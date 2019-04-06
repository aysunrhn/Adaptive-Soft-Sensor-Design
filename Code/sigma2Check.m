function [yqpred, sigma20, predV0, dof0, s2, p2, df, fSigma2, fSigma2EW, age, WSize, paramList] = sigma2Check(XNew, yNew, xq, yq, yqpred, sigma20, predV0, dof0, age, hyperparameter, df, WInit, WSize, paramList, fSigma2, fSigma2EW, qLoc)

% rows: 99.9% / 99.5% / 99% (aParam.alphaFlag)
CBTable(:, :, 1) = [-0.6913, 0.6908; %1.[1:30, 1:30]
                    -0.5866, 0.5863;
                    -0.5369, 0.5369;
                    -0.4063, 0.4059];
CBTable(:, :, 2) = [-0.6913, 0.4992; %2.[1:30, 30:60]
                    -0.5866, 0.4231;
                    -0.5369, 0.3864;
                    -0.3802, 0.2885];
CBTable(:, :, 3) = [-0.4996, 0.6369; %3.[30:60, 1:30]
                    -0.4232, 0.5401;
                    -0.3865, 0.4959;
                    -0.2884, 0.3801]; 
CBTable(:, :, 4) = [-0.4137, 0.4137; %4.[30:60, 30:60]
                    -0.3524, 0.3520;
                    -0.3229, 0.3227;
                    -0.2454, 0.2451];                
s2   = hyperparameter.modelSigma2;
p2   = hyperparameter.Psigma2;
Wmin = paramList.WL;
paramList.sFlag = 1;
if dof0 < 30 % 1 or 2     
    if df < 30 % 1
        lowerBound = CBTable(paramList.alpha, 1, 1);
        upperBound = CBTable(paramList.alpha, 2, 1);
    else % 2
        lowerBound = CBTable(paramList.alpha, 1, 2);
        upperBound = CBTable(paramList.alpha, 2, 2);
    end
else % 3 or 4
    if df < 30 % 3
        lowerBound = CBTable(paramList.alpha, 1, 3);
        upperBound = CBTable(paramList.alpha, 2, 3);
    else % 4
        lowerBound = CBTable(paramList.alpha, 1, 4);
        upperBound = CBTable(paramList.alpha, 2, 4);
    end
end

fSigma2(qLoc + 1)   = log(s2/sigma20);
fSigma2EW(qLoc + 1) = paramList.lambda*fSigma2(qLoc + 1) + (1 - paramList.lambda)*fSigma2EW(qLoc);
paramList.tmpF(qLoc + 1) = fSigma2(qLoc + 1);
paramList.tmpFEW(qLoc + 1) = fSigma2EW(qLoc + 1);
if fSigma2EW(qLoc + 1) > lowerBound
    if fSigma2EW(qLoc + 1) < upperBound
        age = age + 1;
        predV0 = (age - 1)/age*predV0 + 1/age*p2;
        paramList.sFlag = 0;
        return
    else
        paramList.sFlag = 2;
        if ~paramList.f
            % find a new WSize for yn
            paramList.Inc = WSize;
            [yprednew, W, altModel, s2new, p2new, dfnew, fSigma2, fSigma2EW] = findNewWSize(XNew, yNew, ...
                xq, yq, sigma20, qLoc, WInit, WSize, Wmin, fSigma2, fSigma2EW, upperBound, paramList);
            paramList.tmpF(qLoc + 1) = fSigma2(qLoc + 1);
            paramList.tmpFEW(qLoc + 1) = fSigma2EW(qLoc + 1);
            if altModel
                s2 = s2new; p2 = p2new; WSize = W; age = age + 1; df = dfnew;
                yqpred = yprednew;
                paramList.sFlag = 3;
                return
            end
        end
    end
end
sigma20 = s2;
predV0  = p2;
dof0    = df;
age     = 1;
fSigma2(qLoc+1)   = 0;
fSigma2EW(qLoc+1) = 0;
end

function [yqpred, WSize, altModel, s2, p2, df, f2, fEW] = findNewWSize(XNew, yNew, xq, yq, sigma20, qLoc, WInit, WSize, Wmin, f2, fEW, UB, paramList)

tol   = [1e-3 1e-3 1e-3]; lambda = .3; yqpred = NaN(3, 1);
fnew2 = NaN(3, 1); fnewEW = NaN(3, 1); df = NaN(3, 1); s2 = NaN(3, 1); p2 = NaN(3, 1);
WNew  = [max(Wmin, WSize - paramList.Inc*.5) max(Wmin, WSize - paramList.Inc*.2), ...
    WSize, min(WInit, WSize + paramList.Inc*.2)];
sx    = std(XNew); N = length(yNew); 

for i = 1:length(WNew)
    trainIdx = (N - round(WNew(i)) + 1):N;
    Xtrain   = XNew(trainIdx, :); ytrain = yNew(trainIdx); vars = (std(Xtrain)./sx > 1e-3);
    [yhat, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
        xq(:, vars), yq, 'Gaussian', tol);
    yqpred(i) = yhat;
    df(i) = max(WNew(i) - length(find(m)) - 1, 1);
    hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, paramList.sL);
    s2(i) = hyperparameter.modelSigma2;
    p2(i) = hyperparameter.Psigma2;
    fnew2(i)  = log(s2(i)/sigma20);
    fnewEW(i) = lambda*fnew2(i) + (1 - lambda)*fEW(qLoc);
end
[fMin, idx] = nanmin(fnewEW);
if (fMin < UB) && ~isnan(fMin) %(fMin < UB) &&
    altModel = true;
    WSize = round(WNew(idx));
    df = df(idx);
    s2 = s2(idx);
    p2 = p2(idx);
    yqpred = yqpred(idx);
    f2(qLoc + 1)  = fnew2(idx);
    fEW(qLoc + 1) = fnewEW(idx);
    [Xtrain, ytrain, f, WW] = mahalDistCheck(xq, XNew, yNew, WSize, paramList.mahal);
    paramList.f = f; vars = (std(Xtrain)./sx > 1e-3);
    [yhat, m, hyperparameter] = SBPrediction (Xtrain(:, vars), ytrain, ...
        xq(:, vars), yq, 'Gaussian', tol);
    yqpred = yhat;
    WSize = WW;
    df = max(WSize - length(find(m)) - 1, 1);
    hyperparameter.modelSigma2 = max(hyperparameter.modelSigma2, paramList.sL);
    s2 = hyperparameter.modelSigma2;
    p2 = hyperparameter.Psigma2;
    f2(qLoc + 1)  = log(s2/sigma20);
    fEW(qLoc + 1) = lambda*f2(qLoc + 1) + (1 - lambda)*fEW(qLoc);
else
    WSize    = [];
    altModel = false;
end
end
