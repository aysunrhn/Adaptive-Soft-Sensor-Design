function [ypred, err] = MWAdpJITL(X, y, Xtest, ytest, flagM, paramList)
%%% Inputs
% X     : Training set input matrix (predictors), N x p
% y     : Training set output (response), N x 1
% Xtest : Test set input matrix (predictors), N x p
% ytest : Test set output (response), N x 1

%%% flagM =
% 'constMW'   : constant W MW
% 'MWAdp'     : MW_Adp method
% 'MWAdpJITL' : MW_AdpJITL method

%%% paramList =
% - MW parameters
%   paramList.tFlag  : logical, if time indices are used as predictors, they
%                      should be supplied in the first column of both X and Xtest
%   paramList.WSize  : W parameter in constMW / W^U in both MWAdp and MWAdpJITL
%   paramList.WRange : W values for grid search in CV, this parameter is overridden
%                      if WSize is also specificied hence CV is not performed
%   paramList.mahal  : Indices of predictors used in checkM subroutine, leave it
%                      empty to eliminate checkM subroutine
%   paramList.sLFlag : logical, if true sLim is estimated using the training set
%                      if false, sLim = 2.5 x 10^-7
%   paramList.alpha  : Significance level of sigma control chart
%                      1 (99.9%) 2 (99.5%) 3 (99%) 4 (95%)
%   paramList.lambda : lambda value in EWMA
% - CV parameters
% 	paramList.CVType : Type of CV performed to optimize W, 'kfold' is the 
%                      usual kfold CV with M folds and R repetitions, 
%                      'lastblock' leaves rho*N number of observations at 
%                      the end for validation (N is the total # of training
%                      observations)
%   paramList.M      : # of folds in kfold CV
%   paramList.R      : # of repetitions in kfold CV
%   paramList.rho    : validation set size in lastblock CV, 0 < rho < 1
% - JITL parameters
%   paramList.NNType : JITL method type, use 'def' for default (MWAdpJITL) 
%                      and 'lim' to constrain the total training set size (MWAdpJITL (lim))
%   paramList.Wtilde : # of observations in JITL model
%   paramList.TSMax  : Training set size, specify if paramList.NNType = 'lim'
%   paramList.mp     : # of "extra" validation points from the past

%%% Outputs
% ypred : predicted values
% err   : metrics for prediction performance including RMSE, MAE, maxAE, R2

warning('off')
vars = setdiff(1:size(X, 2), find(~range(X)));
if isempty (paramList.WSize) || ~isempty(paramList.WRange)
    switch paramList.CVType
        case 'kfold'
            [~, Wopt] = kfoldCVprogram (X(:, vars), y, paramList);
        case 'lastblock'
            [~, Wopt] = LBCVprogram (X(:, vars), y, paramList);
    end
    paramList.WSize = Wopt;
end
if isempty(paramList.lambda)
    paramList.lambda = 0.3;
end

switch flagM
    case 'constMW'
        ypred = constMW(X(:, vars), y, Xtest(:, vars), ytest, paramList);
    case 'MWAdp'
        ypred = MWAdp(X(:, vars), y, Xtest(:, vars), ytest, paramList);
    case 'MWAdpJITL'
        switch paramList.NNType
            case 'def'
                [ypred, paramList] = JITLsub(X(:, vars), y, Xtest(:, vars), ytest, paramList);
            case 'lim'
                [ypred, paramList] = JITLsubLim(X(:, vars), y, Xtest(:, vars), ytest, paramList);
        end
end
err.RMSE  = sqrt(nanmean((ytest - ypred).^2));
err.MAE   = mean(abs(ytest - ypred));
err.R2    = 1 - sum((ytest - ypred).^2)/(sum((ytest - mean(y)).^2));
err.maxAE = max(abs(ytest - ypred));
end

%% RVM BASED MW MODEL
function [ypred, s2Vec] = constMW(X, y, Xtest, ytest, paramList)

N     = length(y); Ntest = length(ytest);
W     = paramList.WSize; cols = paramList.mahal;
sx    = std(X); XNew = X; yNew = y;
ypred = NaN(Ntest, 1); s2Vec = NaN(Ntest, 1);
tol = [1e-3, 1e-3, 1e-3];
warning('off')
for i = 1:Ntest
    if isempty(cols)
        Xtrain = XNew(end - W + 1:end, :);
        ytrain = yNew(end - W + 1:end);
    else
        [Xtrain, ytrain] = mahalDistCheck(Xtest(i, :), XNew, yNew, W, cols);
    end
    vars = (std(Xtrain)./sx > 1e-3);    
    [yqpred, ~, hyp] = SBPrediction (Xtrain(:, vars), ytrain, Xtest(i, vars), ...
        ytest(i, :), 'Gaussian', tol);
    ypred(i) = yqpred;  s2Vec(i) = max(hyp.modelSigma2, -Inf);
    XNew = [X; Xtest(1:i, :)];
    yNew = [y; ytest(1:i)];
    sx = std(XNew);
end
end

%% KFOLD CV PROGRAM
function [foldError, Wopt, meanError, ICVE] = kfoldCVprogram (X_RL, y_RL, paramList)

N = length(y_RL);
setNo     = paramList.M; permNo = paramList.R; WRange = paramList.WRange;
setSize   = round(N/setNo);
perm      = NaN(N, permNo);
foldError = NaN(setSize, setNo, permNo, length(WRange));
tol = [1e-3, 1e-3, 1e-3];
for p = 1:permNo
    perm(:, p) = randperm (N, N);
    X = X_RL (perm(:, p), :);
    y = y_RL (perm(:, p), :);
    for i = 1:setNo
        testind  = ((i-1)*setSize + 1):(i*setSize);
        testind  = testind(testind <= N);
        trainind = setdiff(1:N, testind);
        for n = 1:length(testind)
            k = 1;
            for j = WRange
                Xtrain = X(trainind((length(trainind) - j + 1):end), :);
                ytrain = y(trainind((length(trainind) - j + 1):end));
                
                yhat = SBPrediction (Xtrain, ytrain, X(testind(n), :), ...
                    y(testind(n)), 'Gaussian', tol);
                foldError(n, i, p, k) = y(testind(n)) - yhat;
                k = k + 1;
            end
            trainind = [trainind, testind(n)];
        end
    end
end
meanError   = nanmean(squeeze(sqrt(nanmean(reshape(foldError.^2, ...
    [size(foldError,1), setNo*permNo, length(SWRange)])))));
[ICVE, ind] = nanmin(meanError(:));
foldError   = reshape(squeeze(foldError(:, :, :, ind)), [setSize, setNo*permNo]);
Wopt = WRange(ind);
% limit      = ICVE + std(sqrt(nanmean(foldError.^2)), 'omitnan')./sqrt(setNo*permNo);
% ind2       = find(meanError < limit, 1);
% Wopt.oneSE = WRange(ind2);
end

%% LAST BLOCK CV PROGRAM
function [errorMat, Wopt, meanError, ICVE] = LBCVprogram (X, y, paramList)

N = length(y_RL);
valRatio = paramList.rho; WRange = paramList.WRange; cols = paramList.mahal;
calSize  = round(N*valRatio);
errorMat = NaN(N - calSize, length(WRange));
trainind = 1:calSize;
testind  = (calSize + 1):N;
tol = [1e-3, 1e-3, 1e-3];
for n = 1:length(testind)
    k = 1;
    for j = WRange
        if isempty(cols)
            Xtrain = X(trainind((length(trainind) - j + 1):end), :);
            ytrain = y(trainind((length(trainind) - j + 1):end));
        else
            [Xtrain, ytrain] = mahalDistCheck(X(testind(n), :), X(trainind, :), y(trainind), j, cols);
        end
        yhat = SBPrediction (Xtrain, ytrain, X(testind(n), :), y(testind(n)), 'Gaussian', tol);
        errorMat(n, k) = y(testind(n)) - yhat;
        k = k + 1;
    end
    trainind = [1:calSize, testind(1:n)];
end
meanError   = nanmean(sqrt(errorMat.^2));
[ICVE, ind] = nanmin(meanError(:));
Wopt        = WRange(ind);
% limit       = ICVE + std(sqrt(errorMat(:, ind).^2), 'omitnan');
% ind2        = find(meanError < limit, 1);
% Wopt.oneSE  = WRange(ind2);
end

%% MWADP
function ypred = MWAdp(X, y, Xtest, ytest, paramList)
N     = length(y); Ntest = length(ytest); XNew = X; yNew = y; 
ypred = NaN(Ntest, 1); 
% WSize settings
WSize = paramList.WSize; WL = paramList.WL; WU = paramList.WSize; cols = paramList.mahal;
% Sigma Check
modelSigma   = NaN(Ntest + 1, 1);
fSigma2      = ones(Ntest + 1, 1); fSigma2EW  = ones(Ntest + 1, 1); 
% RVM Tolerance
tol = [1e-3, 1e-3, 1e-3]; 
trainIdx     = (N-WU+1):N; Xtrain = X(trainIdx, :); ytrain = y(trainIdx);
if paramList.sLFlag
    [~, s2Vec] = constMW(X(1:WU, :), y(1:WU), X(WU + 1:end, :), y(WU + 1:end), paramList);
    sL = median(s2Vec);
else
    sL = 2.5*1e-7;
end
paramList.sL = sL;
for i = 1:Ntest
    sx = std(XNew);
    if i == 1
        W0 = WU;
        [Xtrain, ytrain, f] = mahalDistCheck(Xtest(1, :), XNew, yNew, W0, cols);
        vars = (std(Xtrain)./sx > 1e-3);
        [yqpred, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
            Xtest(i, vars), ytest(i, :), 'Gaussian', tol);
        WHistory(1:2) = W0; WSize = W0; 
        paramList.f = f;
        
        df = max(WHistory(i+1) - length(find(m)) - 1, 1); 
        hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, sL);
        s2 = hyperparameter.modelSigma2;
        p2 = hyperparameter.Psigma2;
        dof0 = df; sigma20 = s2; predV0 = p2;
        fSigma2(1:2)    = 0; age = 1;
        fSigma2EW(1:2)  = 0;
        paramList.tmpF = fSigma2;
        paramList.tmpFEW = fSigma2EW;
    else
        [Xtrain, ytrain, f] = mahalDistCheck(Xtest(i, :), XNew, yNew, WHistory(i+1), cols);
        paramList.f = f;
        vars = (std(Xtrain)./sx > 1e-3);
        [yqpred, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
            Xtest(i, vars), ytest(i, :), 'Gaussian', tol);
        df = max(WHistory(i+1) - length(find(m)) - 1, 1);
        hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, sL);
        s2 = hyperparameter.modelSigma2;
        p2 = hyperparameter.Psigma2;
    end
    if WHistory(i+1) < paramList.WL
        age = age + 1;
    else
        [yqpred, sigma20, predV0, dof0, s2, p2, df, fSigma2, fSigma2EW, age, WSize, paramList] = sigma2Check(XNew, yNew, ...
            Xtest(i, :), ytest(i), yqpred, sigma20, predV0, dof0, age, ...
            hyperparameter, df, WU, WHistory(i+1), paramList, fSigma2, fSigma2EW, i);
    end
    modelSigma(i+1)   = s2;
    WHistory(i+1:i+2) = round(WSize); 
    ypred(i) = yqpred;

    trainIdx = (N + i - round(WSize) + 1):N + i;
    XNew     = [X; Xtest(1:i, :)];
    yNew     = [y; ytest(1:i)];
    Xtrain   = XNew(trainIdx, :);
    ytrain   = yNew(trainIdx);
end
end

%% MWADPJITL PROGRAM
function [ypred, paramList] = JITLsub(X, y, Xtest, ytest, paramList)
N     = length(y); Ntest = length(ytest); XNew = X; yNew = y; 
ypred = NaN(Ntest, 1); 
% WSize settings
WSize = paramList.WSize; WL = paramList.WL; WU = paramList.WSize; cols = paramList.mahal;
% Sigma Check
modelSigma   = NaN(Ntest + 1, 1);
fSigma2      = ones(Ntest + 1, 1); fSigma2EW  = ones(Ntest + 1, 1); 
if paramList.sLFlag
    [~, s2Vec] = constMW(X(1:WU, :), y(1:WU), X(WU + 1:end, :), y(WU + 1:end), paramList);
    sL = median(s2Vec);
else
    sL = 2.5*1e-7;
end
paramList.sL = sL;
% JITL
NNInfo.Wtilde = paramList.Wtilde; NNInfo.N0 = N; NNInfo.tFlag = paramList.tFlag; 
NNInfo.firstIdx = 1; NNInfo.mp = paramList.mp - 1;
% RVM Tolerance
tol = [1e-3, 1e-3, 1e-3]; 
trainIdx = (N - WU + 1):N; Xtrain = X(trainIdx, :); ytrain = y(trainIdx);
for i = 1:Ntest
    sx = std(XNew);
    if i == 1
        W0 = WU; MW = WU;
        [Xtrain, ytrain, f, WW] = mahalDistCheck(Xtest(1, :), XNew, yNew, W0, cols);
        vars = (std(Xtrain)./sx > 1e-3);
        [yqpred, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
            Xtest(i, vars), ytest(i, :), 'Gaussian', tol);
        WHistory(1:2) = W0; WSize = W0; 
        paramList.f = f;
        if f
            MW = WW;
        end
        df = max(WHistory(i+1) - length(find(m)) - 1, 1); 
        hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, sL);
        s2 = hyperparameter.modelSigma2;
        p2 = hyperparameter.Psigma2;
        dof0 = df; sigma20 = s2; predV0 = p2;
        fSigma2(1:2)    = 0; age = 1;
        fSigma2EW(1:2)  = 0;
        paramList.tmpF = fSigma2;
        paramList.tmpFEW = fSigma2EW;
    else
        [Xtrain, ytrain, f, WW] = mahalDistCheck(Xtest(i, :), XNew, yNew, WHistory(i+1), cols);
        paramList.f = f;
        vars = (std(Xtrain)./sx > 1e-3);
        [yqpred, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
            Xtest(i, vars), ytest(i, :), 'Gaussian', tol);
        df = max(WHistory(i+1) - length(find(m)) - 1, 1);
        hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, sL);
        s2 = hyperparameter.modelSigma2;
        p2 = hyperparameter.Psigma2;
    end
    if WHistory(i+1) < paramList.WL
        age = age + 1;
    else
        [yqpred, sigma20, predV0, dof0, s2, p2, df, fSigma2, fSigma2EW, age, WSize, paramList] = sigma2Check(XNew, yNew, ...
            Xtest(i, :), ytest(i), yqpred, sigma20, predV0, dof0, age, ...
            hyperparameter, df, WU, WHistory(i+1), paramList, fSigma2, fSigma2EW, i);
    end
    modelSigma(i+1)   = s2;
    WHistory(i+1:i+2) = round(WSize); 
    ypredMonitor(i) = yqpred;
    NNInfo.yhatMW   = ypredMonitor(i);
    if paramList.f
        MW = WW;
    else
        MW = WSize;
    end
    trainIdx = (N + i - round(WSize) + 1):N + i;
    
    if i > NNInfo.mp + 1
        candIdx = setdiff(1:N + i - 1, (N + i - 1 - max(MW, WU) + 1):N + i - 1);
        if length(candIdx) < paramList.Wtilde
            ypredParallel(i) = ypredMonitor(i);
        else
            [yqpred, ~, ~, ~, NNInfo] = buildJITL(XNew, yNew, Xtest(1:i, :), ...
                ytest(1:i), i-1, [WU MW WL paramList.Wtilde], NNInfo, candIdx, paramList);
            ypredParallel(i) = yqpred;
        end
    end
    XNew     = [X; Xtest(1:i, :)];
    yNew     = [y; ytest(1:i)];
    Xtrain   = XNew(trainIdx, :);
    ytrain   = yNew(trainIdx);
end
ypred = [ypredMonitor(1:NNInfo.mp + 1)'; ypredParallel(NNInfo.mp + 2:end)'];
end

%% MWAADPJITL (lim) PROGRAM
function [ypred, paramList] = JITLsubLim(X, y, Xtest, ytest, paramList)
N     = length(y); Ntest = length(ytest); XNew = X; yNew = y; 
ypred = NaN(Ntest, 1); 
% WSize settings
WSize = paramList.WSize; WL = paramList.WL; WU = paramList.WSize; cols = paramList.mahal;
% Sigma Check
modelSigma   = NaN(Ntest + 1, 1);
fSigma2      = ones(Ntest + 1, 1); fSigma2EW  = ones(Ntest + 1, 1); 
if paramList.sLFlag
    [~, s2Vec] = constMW(X(1:WU, :), y(1:WU), X(WU + 1:end, :), y(WU + 1:end), paramList);
    sL = median(s2Vec);
else
    sL = 2.5*1e-7;
end
paramList.sL = sL;
% JITL
NNInfo.Wtilde = paramList.Wtilde; NNInfo.N0 = N; NNInfo.TS0 = N; 
NNInfo.firstIdx = 1; NNInfo.mp = paramList.mp - 1; NNInfo.tFlag = paramList.tFlag; 
if isempty(paramList.TSMax)
    NNInfo.TSMax = N;
else
    NNInfo.TSMax = paramList.TSMax;
end
% RVM Tolerance
tol = [1e-3, 1e-3, 1e-3]; 
trainIdx     = (N-WU+1):N; Xtrain = X(trainIdx, :); ytrain = y(trainIdx);

for i = 1:Ntest
    sx = std(XNew);
    if i == 1
        W0 = WU; MW = WU;
        [Xtrain, ytrain, f, WW] = mahalDistCheck(Xtest(1, :), XNew, yNew, W0, cols);
        vars = (std(Xtrain)./sx > 1e-3);
        [yqpred, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
            Xtest(i, vars), ytest(i, :), 'Gaussian', tol);
        WHistory(1:2) = W0; WSize = W0; 
        paramList.f = f;
        if f
            MW = WW;
        end
        df = max(WHistory(i+1) - length(find(m)) - 1, 1); 
        hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, sL);
        s2 = hyperparameter.modelSigma2;
        p2 = hyperparameter.Psigma2;
        dof0 = df; sigma20 = s2; predV0 = p2;
        fSigma2(1:2)    = 0; age = 1;
        fSigma2EW(1:2)  = 0;
        paramList.tmpF = fSigma2;
        paramList.tmpFEW = fSigma2EW;
    else
        [Xtrain, ytrain, f, WW] = mahalDistCheck(Xtest(i, :), XNew, yNew, WHistory(i+1), cols);
        paramList.f = f;
        vars = (std(Xtrain)./sx > 1e-3);
        [yqpred, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
            Xtest(i, vars), ytest(i, :), 'Gaussian', tol);
        df = max(WHistory(i+1) - length(find(m)) - 1, 1);
        hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, sL);
        s2 = hyperparameter.modelSigma2;
        p2 = hyperparameter.Psigma2;
    end
    if WHistory(i+1) < paramList.WL
        age = age + 1;
    else
        [yqpred, sigma20, predV0, dof0, s2, p2, df, fSigma2, fSigma2EW, age, WSize, paramList] = sigma2Check(XNew, yNew, ...
            Xtest(i, :), ytest(i), yqpred, sigma20, predV0, dof0, age, ...
            hyperparameter, df, WU, WHistory(i+1), paramList, fSigma2, fSigma2EW, i);
    end
    modelSigma(i+1)   = s2;
    WHistory(i+1:i+2) = round(WSize); 
    ypredMonitor(i) = yqpred;
    NNInfo.yhatMW   = ypredMonitor(i);
    if paramList.f
        MW = WW;
    else
        MW = WSize;
    end
    if i > NNInfo.mp + 1
        candIdx = setdiff(1:N - 1, (N - 1 - max(MW, WU) + 1):N - 1);
        if length(candIdx) < paramList.Wtilde
            ypredParallel(i) = ypredMonitor(i);
        else
            [yqpred, ~, ~, ~, NNInfo] = buildJITL(XNew, yNew, Xtest(1:i, :), ...
                ytest(1:i), i-1, [WU MW WL paramList.Wtilde], NNInfo, candIdx, paramList);
            ypredParallel(i) = yqpred;
        end
    end
    XNew     = [XNew(max(1, N - NNInfo.TSMax + 2):end, :); Xtest(i, :)];
    yNew     = [yNew(max(1, N - NNInfo.TSMax + 2):end, :); ytest(i)];
    N        = length(yNew);
    trainIdx = (N - round(WSize) + 1):N;
    Xtrain   = XNew(trainIdx, :);
    ytrain   = yNew(trainIdx);
    NNInfo.N0 = N - i;
end
ypred = [ypredMonitor(1:NNInfo.mp + 1)'; ypredParallel(NNInfo.mp + 2:end)'];
end
