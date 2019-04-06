function [yqpred, trainMW, trainNN, WSize, NNInfo] = buildJITL(XNew, yNew, Xtest, ytest, qLoc, WH, NNInfo, candIdx, paramList)
% {XNew, yNew} contains the first 1:N + i - 1 points (same as during the prediction of xq(i))
% {Xtest, ytest} contains the first i test points
% WH: the window size assigned for xq(i) and xq(i+1)
% dmax: average distance of Xtrain(i) from xq(i)
% candIdx: everything except for trainIdx(i)
vv = find(range(XNew)); 
[xxsc, mxx, sxx] = auto(XNew(candIdx, vv)); xqsc = scale(Xtest(end-1, vv), mxx, sxx);
if NNInfo.tFlag
    col1 = 2;
else
    col1 = 1;
end
tol = [1e-3 1e-3 1e-3];
switch paramList.NNType
    case 'def'
        N = length(yNew) - qLoc; sx = std(XNew(1:end-1, :));
        trainMW = (N + qLoc - WH(2)):N + qLoc - 1;
        NNInfo.w = 1;
        NNInfo.trainNN = [];
        NNInfo.flag = false;
        if length(WH) > 3
            deltaSize = WH(4);
            if isnan(deltaSize)
               deltaSize = max(NNInfo.Wtilde, 10); 
            end
        else
            deltaSize = WH(1) - WH(2);
            deltaSize = min(length(candIdx), deltaSize);
            deltaSize = 2*ceil((deltaSize - 1)/2) + 1;
        end
        upperLim  = round(deltaSize*.75);
        slideW    = ceil(deltaSize/5);
        
        d = pdist2(xxsc(:, col1:end), xqsc(:, col1:end));
        dmax = median(d) - mad(d, 1)/2;
        weight = exp(-(-deltaSize*0.5:1:deltaSize*0.5).^2/(5*deltaSize));
        weight = weight./sum(weight);
        dFilter = filter(weight, 1, d);
        dFilter(1:deltaSize-1) = dFilter(1:deltaSize-1)./[(1:deltaSize-1)']*deltaSize;
        dFilter(1:round(deltaSize*.5)) = [];
        idxOK = find(dFilter < dmax);
        if isempty(idxOK)
            trainNN = [];
            yqpred = NNInfo.yhatMW;
            WSize   = length([trainNN, trainMW]);
            NNInfo.PENmw = NaN;
            NNInfo.PENnn = NaN;
            return
        end
        D = diff(idxOK);
        newClassRaw = find(D > deltaSize*0.5);
        newClass = idxOK([1 newClassRaw'+1]);
        newClassBound = idxOK(newClassRaw); newClassBound(end+1) = idxOK(end);
        noClass  = length(newClass);
        for i = 1:noClass
            baseIdx{i} = max(newClass(i)-round(deltaSize/2) + 1, 1):...
                min(newClassBound(i) + round(deltaSize/2), candIdx(end));
            if length(baseIdx{i}) < max(10, upperLim)
                allTrainIdx{i, 1}  = [];
                clusterPE(i, 1)    = NaN;
                clusterSigma(i, 1) = NaN;
                continue
            else
                noModels = floor((length(baseIdx{i}) - deltaSize)/slideW) + 1;
                a = lagmatrix(baseIdx{i}, 0:-1:-deltaSize+1);
                if noModels < 1
                    modelIdx = a(1, 1:length(baseIdx{i}));
                    noModels = 1;
                else
                    modelIdx = a(1:slideW:noModels*slideW, :);
                end
            end
            for j = 1:noModels
                newIdx   = modelIdx(j, :);
                trainIdx = unique(sort(newIdx(end - min(deltaSize, length(baseIdx{i})) + 1:end)));
                vars = find(std(XNew(trainIdx, :))./sx > 1e-3);
                [yhat, mu, hyp] = SBPrediction (XNew(trainIdx, vars(col1:end)), yNew(trainIdx), ...
                    Xtest(end-1, vars(col1:end)), ytest(end-1, :), 'Gaussian', tol);
                allTrainIdx{i, j}  = trainIdx;
                clusterPE(i, j, 1) = ytest(end-1) - yhat;
                clusterSigma(i, j) = hyp.modelSigma2;
                if ~isempty(NNInfo.firstIdx)
                    NNInfo.mu = mu;
                    if NNInfo.tFlag
                        NNInfo.vars = vars(col1:end)-col1+1;
                    else
                        NNInfo.vars = vars;
                    end
                    prevPE = backToTheFuture(trainIdx, XNew(:, col1:end), yNew, qLoc, NNInfo);
                    clusterPE(i, j, 2:2+size(prevPE, 1)-1) = prevPE;
                end
            end
        end
                
        % find the SW prediction of the current point
        vars = std(XNew(trainMW, :))./sx > 1e-3;
        [yhat, ~, hyp] = SBPrediction (XNew(trainMW, vars), yNew(trainMW), ...
            Xtest(end-1, vars), ytest(end-1, :), 'Gaussian', tol);
        clusterPE(end+1, 1, 1)  = ytest(end-1) - yhat;
        clusterSigma(end+1, 1)  = hyp.modelSigma2;
        if ~isempty(NNInfo.firstIdx)
            NNInfo.flag = true;
            prevPE = backToTheFuture(trainMW, XNew, yNew, qLoc, NNInfo);
            clusterPE(end, 1, 2:2+size(prevPE, 1)-1) = prevPE;
        end
        
        clusterPE(clusterPE == 0) = NaN;
        clusterSigma(clusterSigma == 0) = NaN;
        weightedPE    = weightedSum(clusterPE.^2, clusterSigma, 1, 'off');
        minimizeThis  = weightedPE(1:end-1, :);
        [minVal, idx] = min(minimizeThis(:));
        if isnan(minVal)
            s1 = size(weightedPE, 1); s2 = 1;
            trainNN  = [];
        else
            [s1, s2] = ind2sub(size(minimizeThis), idx);
            trainNN  = allTrainIdx{s1, s2};
        end        
        WSize = length(trainNN);
        
        % find the MW prediction of the next point
        trainMW = [trainMW(2:end), N + qLoc];
        Xtrain  = XNew(trainMW, :);
        ytrain  = yNew(trainMW);
        vars = (std(Xtrain)./sx > 1e-3);
        yhatMW = NNInfo.yhatMW;
        
        if isempty (trainNN)
            yhatNN = yhatMW;
        else
            % find the NN prediction of the next point
            Xtrain = XNew(trainNN, :);
            ytrain = yNew(trainNN);
            sx     = std(XNew); vars = find(std(Xtrain)./sx > 1e-3);
            yhatNN = SBPrediction (Xtrain(:, vars(col1:end)), ytrain, ...
                Xtest(end, vars(col1:end)), ytest(end, :), 'Gaussian', tol);
        end
        weight = 1./[weightedPE(end, 1), weightedPE(s1, s2)];
        weight = weight(1)/sum(weight); 
        yqpred = weight*yhatMW + (1 - weight)*yhatNN;
        NNInfo.w = weight;
        NNInfo.trainNN = trainNN;        
    case 'lim'
        N = length(yNew); sx = std(XNew(1:end-1, :));
        trainMW = (N - WH(2)):N - 1;
        NNInfo.w = 1;
        NNInfo.trainNN = [];
        NNInfo.flag = false;
        if length(WH) > 3
            deltaSize = WH(4);
            if isnan(deltaSize)
               deltaSize = max(NNInfo.Wtilde, 10); 
            end
        else
            deltaSize = WH(1) - WH(2);
            deltaSize = min(length(candIdx), deltaSize);
            deltaSize = 2*ceil((deltaSize - 1)/2) + 1;
        end
        upperLim  = round(deltaSize*.75);
        slideW    = ceil(deltaSize/5);
        
        d = pdist2(xxsc(:, col1:end), xqsc(:, col1:end));
        dmax = median(d) - mad(d, 1)/2;
        weight = exp(-(-deltaSize*0.5:1:deltaSize*0.5).^2/(5*deltaSize));
        weight = weight./sum(weight);
        dFilter = filter(weight, 1, d);
        dFilter(1:deltaSize-1) = dFilter(1:deltaSize-1)./[(1:deltaSize-1)']*deltaSize;
        dFilter(1:round(deltaSize*.5)) = [];
        idxOK = find(dFilter < dmax);
        if isempty(idxOK)
            trainNN = [];
            yqpred = NNInfo.yhatMW;
            WSize   = length([trainNN, trainMW]);
            NNInfo.PENmw = NaN;
            NNInfo.PENnn = NaN;
            return
        end
        D = diff(idxOK);
        newClassRaw = find(D > deltaSize*0.5);
        newClass = idxOK([1 newClassRaw'+1]);
        newClassBound = idxOK(newClassRaw); newClassBound(end+1) = idxOK(end);
        noClass  = length(newClass);
        for i = 1:noClass
            baseIdx{i} = max(newClass(i)-round(deltaSize/2) + 1, 1):...
                min(newClassBound(i) + round(deltaSize/2), candIdx(end));
            if length(baseIdx{i}) < max(10, upperLim)
                allTrainIdx{i, 1}  = [];
                clusterPE(i, 1)    = NaN;
                clusterSigma(i, 1) = NaN;
                continue
            else
                noModels = floor((length(baseIdx{i}) - deltaSize)/slideW) + 1;
                a = lagmatrix(baseIdx{i}, 0:-1:-deltaSize+1);
                if noModels < 1
                    modelIdx = a(1, 1:length(baseIdx{i}));
                    noModels = 1;
                else
                    modelIdx = a(1:slideW:noModels*slideW, :);
                end
            end
            for j = 1:noModels
                newIdx   = modelIdx(j, :);
                trainIdx = unique(sort(newIdx(end - min(deltaSize, length(baseIdx{i})) + 1:end)));
                vars = find(std(XNew(trainIdx, :))./sx > 1e-3);
                [yhat, mu, hyp] = SBPrediction (XNew(trainIdx, vars(col1:end)), yNew(trainIdx), ...
                    Xtest(end-1, vars(col1:end)), ytest(end-1, :), 'Gaussian', tol);
                allTrainIdx{i, j}  = trainIdx;
                clusterPE(i, j, 1) = ytest(end-1) - yhat;
                clusterSigma(i, j) = hyp.modelSigma2;
                if ~isempty(NNInfo.firstIdx)
                    NNInfo.mu = mu;
                    if NNInfo.tFlag
                        NNInfo.vars = vars(col1:end)-col1+1;
                    else
                        NNInfo.vars = vars;
                    end
                    prevPE = backToTheFuture(trainIdx, XNew(:, col1:end), yNew, qLoc, NNInfo);
                    clusterPE(i, j, 2:2+size(prevPE, 1)-1) = prevPE;
                end
            end
        end
               
        % find the SW prediction of the current point
        vars = std(XNew(trainMW, :))./sx > 1e-3;
        [yhat, ~, hyp] = SBPrediction (XNew(trainMW, vars), yNew(trainMW), ...
            Xtest(end-1, vars), ytest(end-1, :), 'Gaussian', tol);
        clusterPE(end+1, 1, 1)  = ytest(end-1) - yhat;
        clusterSigma(end+1, 1)  = hyp.modelSigma2;
        if ~isempty(NNInfo.firstIdx)
            NNInfo.flag = true;
            prevPE = backToTheFuture(trainMW, XNew, yNew, qLoc, NNInfo);
            clusterPE(end, 1, 2:2+size(prevPE, 1)-1) = prevPE;
        end
        
        clusterPE(clusterPE == 0) = NaN;
        clusterSigma(clusterSigma == 0) = NaN;
        weightedPE    = weightedSum(clusterPE.^2, clusterSigma, 1, 'off');
        minimizeThis  = weightedPE(1:end-1, :);
        [minVal, idx] = min(minimizeThis(:));
        if isnan(minVal)
            s1 = size(weightedPE, 1); s2 = 1;
            trainNN  = [];
        else
            [s1, s2] = ind2sub(size(minimizeThis), idx);
            trainNN  = allTrainIdx{s1, s2};
        end
        WSize = length(trainNN);
        
        % find the MW prediction of the next point
        trainMW = [trainMW(2:end), N];
        Xtrain  = XNew(trainMW, :);
        ytrain  = yNew(trainMW);
        vars = (std(Xtrain)./sx > 1e-3);
        yhatMW = NNInfo.yhatMW;
        errMW   = abs(ytest(end) - yhatMW);
        
        if isempty (trainNN)
            yhatNN = yhatMW;
        else
            % find the NN prediction of the next point
            Xtrain = XNew(trainNN, :);
            ytrain = yNew(trainNN);
            sx     = std(XNew); vars = find(std(Xtrain)./sx > 1e-3);
            yhatNN = SBPrediction (Xtrain(:, vars(col1:end)), ytrain, ...
                Xtest(end, vars(col1:end)), ytest(end, :), 'Gaussian', tol);
        end
        weight = 1./[weightedPE(end, 1), weightedPE(s1, s2)];
        weight = weight(1)/sum(weight); 
        yqpred = weight*yhatMW + (1 - weight)*yhatNN;
        NNInfo.w = weight;
        NNInfo.trainNN = trainNN;        
end
end
%% Back to the future
function prevPE = backToTheFuture(trainIdx, X, y, qLoc, NNInfo)

tol = [1e-3 1e-3 1e-3];
maxPast = NNInfo.mp;
prevIdx = qLoc-1:-1:NNInfo.firstIdx;
prevIdx = prevIdx(1:min(length(prevIdx), maxPast));
prevPE  = NaN(length(prevIdx), 1);

for i = 1:length(prevIdx)
    xq = X(NNInfo.N0 + prevIdx(i), :);
    yq = y(NNInfo.N0 + prevIdx(i));
    removeIdx = NNInfo.N0 + (prevIdx(i):qLoc-1);
    XNew   = X(1:NNInfo.N0 + prevIdx(i) - 1, :);
    yNew   = y(1:NNInfo.N0 + prevIdx(i) - 1);   
    addIdx = max(trainIdx(find(diff(trainIdx) > 1) + 1)-1, 1);
    if max(trainIdx) >= NNInfo.N0 + prevIdx(i)
        % this must be MW
        addIdx = max(trainIdx(1) - length(removeIdx), 1):max(trainIdx(1) - 1, 1);
    end
    newIdx = min(unique([setdiff(trainIdx, removeIdx), addIdx]), NNInfo.N0 + prevIdx(i)-1);
    Xtrain = XNew(newIdx, :);
    ytrain = yNew(newIdx);
    if NNInfo.flag
        sx     = std(XNew); vars = std(Xtrain)./sx > 1e-3;
        yhat   = SBPrediction (Xtrain(:, vars), ytrain, ...
            xq(:, vars), yq, 'Gaussian', tol);
    else
        [~, mx, sx] = auto(Xtrain(:, NNInfo.vars));
        xqsc = (xq(:, NNInfo.vars) - mx)./sx;
        yhat = xqsc*NNInfo.mu*std(ytrain) + mean(ytrain);
    end
    prevPE(i) = yq - yhat;
end
end

%% Weighted Sum
function minimizeThis = weightedSum(clusterPE, clusterSigma, relWeight, flagsc)

numPoints = size(clusterPE, 3);
wVector = .3*.7.^[0:numPoints - 1]./sum(.3*.7.^[0:numPoints - 1]);
switch flagsc
    case 'on'
        wScaled = wVector./median(reshape(clusterPE, ...
                [size(clusterPE,1)*size(clusterPE,2), size(clusterPE, 3)]), 'omitnan');
    case 'off'
        wScaled = wVector;
end
wMatrix = reshape(repelem(wScaled, size(clusterPE, 1), size(clusterPE, 2)), ...
            size(clusterPE));
weightedPE = nansum(clusterPE.*wMatrix, 3);
minimizeThis = relWeight*weightedPE + (1-relWeight)*clusterSigma;
minimizeThis(minimizeThis == 0) = NaN;
end

%% Ensemble Predictions
function ypred = ensemblePrediction(trainIdx, X, y, Xtest, ytest)

tol  = [1e-3 1e-3 1e-3];
XNew = [X; Xtest(end-1, :)];
yNew = [y; ytest(end-1)];

Xtrain = XNew(trainIdx, :);
ytrain = yNew(trainIdx);
vars = (std(Xtrain)./std(XNew) > 1e-3);
yhat   = SBPrediction (Xtrain(:, vars), ytrain, ...
    Xtest(end, vars), ytest(end, :), 'Gaussian', tol);
ypred  = yhat;
end