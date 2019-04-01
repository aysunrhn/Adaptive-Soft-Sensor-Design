function CSTRData_PLRPaper(cond, TY, TX, vFlag, N, modelList, WRange, simNo)
%%%
cd 'C:\Users\ChE\Google Drive\CMPE 547\CSTR\Simulink'
switch cond
    case 'COND1'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond1\COND1_50Runs_Ts2min.mat';
    case 'COND1b'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond1\COND1b_50Runs_Ts2min.mat';
    case 'COND1bRamp'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond1\COND1bRamp_20Runs_Ts2min.mat';
    case 'COND2'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond2\COND2_50Runs_Ts2min.mat';
    case 'COND2b'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond2\COND2b_50Runs_Ts2min.mat';
    case 'COND2Ramp'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond2\COND2Ramp_50Runs_Ts2min.mat';   
    case 'COND2bRamp'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond2\COND2bRamp_50Runs_Ts2min.mat';
    case 'COND2StepRamp'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond2\COND2StepRamp_20Runs_Ts2min.mat';
    case 'COND2pureRamp'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Cond2\COND2pureRamp_20Runs_Ts2min.mat';
    case 'DEF'
        simFile = 'C:\Users\ChE\Google Drive\Long Ass Simulations\Case 1\Default\DEFAULT20Runs_Ts2min.mat';
end
load(simFile)
dd = lcm(TY*60/TX, round(TX/2));
lastRow = mod(length(YCell{1})-1, dd);
skipTest = 60/TX*TY;
trainSet = (300-N + 1):300;
for i = simNo
    [Xtrain, ytrain, Xtest, ytest] = bigTimeConversion({YCell{i}(1:end-lastRow)}, {fayCell{i}(1:end-lastRow, :)}, 2, TX, 10, TY, 300, 'avg');
    XtrainC{i} = Xtrain;
    Xtest      = Xtest(1:skipTest:end, :);
    XtestC{i}  = Xtest(1:400, :);
    ytrainC{i} = ytrain;
    ytest      = ytest(1:skipTest:end);
    ytestC{i}  = ytest(1:400);
end

fname = ['C:\Users\ChE\Google Drive\Paper\Post-review\CSTR\', ...
    cond, '_Lag', num2str(vFlag), 'bizimkiTurev.mat'];

M = 10; R = 20;
SD = [81473, 90579, 2699, 91335, 63234, 9754, 27849, 54685, 95744, 96481, ...
    15760,	97049, 95706, 48532, 80017, 14187, 42170, 91558, 79207, 95932];
if ~exist(fname, 'file')
    save(fname, 'SD')
else
    load(fname)
end
switch vFlag
    case 'vSel'
        pSel = [2:5, 7, 8, 10, 14, 15, 17];
    otherwise
        pSel = 1:vFlag*19;
end
for m = 1:length(modelList)
    switch char(modelList(m))
        case 'MW-TS'
            %%
            cd 'C:\Users\ChE\Google Drive\Local Methods'
            % 1.1 PLS MW (TS)
                PLSParam.method = 'offline';
                PLSParam.flagMin = 'min'; PLSParam.flagCV = 'kfold';
                PLSParam.M = 10; PLSParam.R = 20;
                PLSParam.sRatio = []; PLSParam.bufferRatio = [];
                PLSParam.lowerLim = 1; PLSParam.upperLim = min(25, length(pSel));
                PLSParam.cols = [];
                for i = 1:length(WRange)
                    PLSParam.SWRange = WRange(i);
                    parfor j = simNo
                        rng(SD(j))
                        Xtrain = XtrainC{j};
                        Xtest  = XtestC{j};
                        ytrain = ytrainC{j};
                        ytest  = ytestC{j};
                        [ypred, RMSEpred, R2pred, Lopt, ~, tt] = adaptivePLSPrediction(Xtrain(trainSet, pSel), ...
                            ytrain(trainSet), Xtest(:, pSel), ytest, PLSParam);
                        YC{j} = ypred.int;
                        RC{j} = RMSEpred;
                        R2C{j} = R2pred;
                        LC{j} = Lopt;
                        TC{j} = tt;
                        
                    end
                    
                    for j = simNo
                        PLS.ypred(:, i, j) = YC{j};
                        PLS.RMSE(i, j)     = RC{j}.int;
                        PLS.MAE(i, j)      = RC{j}.MAE;
                        PLS.R2(i, j)       = R2C{j}.int;
                        PLS.Lopt(i, j)     = LC{j};
                        PLS.ETime(i, j)    = mean(TC{j}.timeCount(2:end));
                        PLS.avgVIF(i, j)   = TC{j}.avgVIF;
                        sortedErr = sort(abs(ytestC{j} - YC{j}));
                        PLS.maxAE(i, j)    = max(sortedErr);
                        PLS.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                        PLS.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                    end
                    clear PC YC RC R2C LC VC TC
                    save(fname, 'PLS', '-append')
                    display(['W = ', num2str(WRange(i)), ' | PLS_MW^TS RMSE (0): ', num2str(mean(PLS.RMSE(i, :), 2))])
                end
            
            % Lasso MW (TS)
                LassoParam.flagM   = 'vanilla'; LassoParam.method  = 'offline';
                LassoParam.flagE   = 'min'; LassoParam.flagCV = 'kfold';
                LassoParam.M = 10; LassoParam.R = 20;
                LassoParam.sRatio  = []; LassoParam.bufferRatio = [];
                LassoParam.cols = [];
                for i = 1:length(WRange)
                LassoParam.SWRange = WRange(i); 
                parfor j = simNo
                    rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, vs, lmb, beta, ~, tt] = adaptiveLassoPrediction(Xtrain(trainSet, pSel),...
                        ytrain(trainSet), Xtest(:, pSel), ytest, LassoParam);                 
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    LC{j} = lmb;
                    VC{j} = vs;
                    TC{j} = tt;
                end
                for j = simNo
                    Lasso.ypred(:, i, j) = YC{j};
                    Lasso.RMSE(i, j)     = RC{j}.int;
                    Lasso.MAE(i, j)      = RC{j}.MAE;
                    Lasso.R2(i, j)       = R2C{j}.int;
                    Lasso.lambda(i, j)   = LC{j};
                    Lasso.vs{i, j}       = VC{j};
                    Lasso.ETime(i, j)    = mean(TC{j}.timeCount(2:end));
                    Lasso.avgVIF(i, j)   = TC{j}.avgVIF;
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    Lasso.maxAE(i, j)    = sortedErr(end);
                    Lasso.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    Lasso.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                clear PC YC RC R2C LC VC TC
                save(fname, 'Lasso', '-append')
                display(['W = ', num2str(WRange(i)), ' | Lasso_MW^TS RMSE (0): ', num2str(mean(Lasso.RMSE(i, :), 2))])
            end
            
            % RVM MW 
            flagA = 'SW'; aParam.tuneSW = 'off'; aParam.flagInc = [];
            aParam.stdPEflag = false; aParam.timeFlag = 'false';
            aParam.cols = [];
            for i = 1:length(WRange)
                aParam.WSize = WRange(i); aParam.WRange = aParam.WSize;
                parfor j = simNo
                     rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, ~, ~, ~, ~, tt] = adaptiveSBPrediction(Xtrain(trainSet, pSel),...
                        ytrain(trainSet), Xtest(:, pSel), ytest, aParam, flagA);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    TC{j} = tt;
                    
                end
                
                for j = simNo
                    RVM.ypred(:, i, j) = YC{j};
                    RVM.RMSE(i, j)     = RC{j}.int;
                    RVM.R2(i, j)       = R2C{j}.int;
                    RVM.ETime(i, j)    = mean(TC{j}.timeCounter);
                    RVM.avgVIF(i, j)   = TC{j}.avgVIF;
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    RVM.MAE(i, j)      = mean(sortedErr);
                    RVM.maxAE(i, j)    = sortedErr(end);
                    RVM.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    RVM.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                 clear PC YC RC R2C LC VC TC
                save(fname, 'RVM', '-append')
                display(['W = ', num2str(WRange(i)), ' | RVM_MW RMSE (0): ', num2str(mean(RVM.RMSE(i, :), 2))])
           end
        case 'MW-W'
            %%
            cd 'C:\Users\ChE\Google Drive\Local Methods'
            % 1.1 PLS MW (W)
            PLSParam.method = 'SW';
            PLSParam.flagMin = 'min'; PLSParam.flagCV = 'kfold';
            PLSParam.M = 10; PLSParam.R = 1;
            PLSParam.sRatio = []; PLSParam.bufferRatio = [];
            PLSParam.lowerLim = 1; PLSParam.upperLim = min(25, length(pSel));
            PLSParam.cols = [];
            for i = 1:length(WRange)
                PLSParam.SWRange = WRange(i);
                parfor j = simNo
                    rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, Lopt, ~, tt] = adaptivePLSPrediction(Xtrain(trainSet, pSel), ...
                        ytrain(trainSet), Xtest(:, pSel), ytest, PLSParam);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    LC{j} = Lopt;
                    TC{j} = tt;
                    
                end
                
                for j = simNo
                    PLSOn.ypred(:, i, j) = YC{j};
                    PLSOn.RMSE(i, j)     = RC{j}.int;
                    PLSOn.MAE(i, j)      = RC{j}.MAE;
                    PLSOn.R2(i, j)       = R2C{j}.int;
                    PLSOn.Lopt(:, i, j)     = LC{j};
                    PLSOn.ETime(i, j)    = mean(TC{j}.timeCount(2:end));
                    PLSOn.avgVIF(i, j)   = TC{j}.avgVIF;
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    PLSOn.maxAE(i, j)    = max(sortedErr);
                    PLSOn.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    PLSOn.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                clear PC YC RC R2C LC VC TC
                save(fname, 'PLSOn', '-append')
                display(['W = ', num2str(WRange(i)), ' | PLS_MW^W RMSE (0): ', num2str(mean(PLSOn.RMSE(i, :), 2))])
            end
            
            % Lasso MW (TS)
            LassoParam.flagM   = 'vanilla'; LassoParam.method  = 'SW';
            LassoParam.flagE   = 'min'; LassoParam.flagCV = 'kfold';
            LassoParam.M = 10; LassoParam.R = 1;
            LassoParam.sRatio  = []; LassoParam.bufferRatio = [];
            LassoParam.cols = [];
            for i = 1:length(WRange)
                LassoParam.SWRange = WRange(i);
                parfor j = simNo
                    rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, vs, lmb, beta, ~, tt] = adaptiveLassoPrediction(Xtrain(trainSet, pSel),...
                        ytrain(trainSet), Xtest(:, pSel), ytest, LassoParam);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    LC{j} = lmb;
                    VC{j} = vs;
                    TC{j} = tt;
                end
                for j = simNo
                    LassoOn.ypred(:, i, j) = YC{j};
                    LassoOn.RMSE(i, j)     = RC{j}.int;
                    LassoOn.MAE(i, j)      = RC{j}.MAE;
                    LassoOn.R2(i, j)       = R2C{j}.int;
                    LassoOn.lambda(:, i, j)   = LC{j};
                    LassoOn.vs{i, j}       = VC{j};
                    LassoOn.ETime(i, j)    = mean(TC{j}.timeCount(2:end));
                    LassoOn.avgVIF(i, j)   = TC{j}.avgVIF;
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    LassoOn.maxAE(i, j)    = sortedErr(end);
                    LassoOn.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    LassoOn.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                clear PC YC RC R2C LC VC TC
                save(fname, 'LassoOn', '-append')
                display(['W = ', num2str(WRange(i)), ' | Lasso_MW^W RMSE (0): ', num2str(mean(LassoOn.RMSE(i, :), 2))])
            end

        case 'batch'
            %%
            cd 'C:\Users\ChE\Google Drive\Local Methods'
            % Lasso Batch
            Lags = [1:10]*19;
            pArray = 1:9;
            for i = pArray
                pSel = 1:Lags(i);
                parfor j = simNo
                    rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, vs, lmb, beta] = lassoPrediction(Xtrain(trainSet, pSel), ...
                        ytrain(trainSet), Xtest(:, pSel), ytest, [], [], [], 'kfold', 'min', 'vanilla', ...
                        10, 20, [], []);
                    YC{j}  = ypred.int;
                    RC{j}  = RMSEpred;
                    R2C{j} = R2pred;
                    LC{j}  = lmb;
                    VC{j}  = vs;
                    BB{j}  = beta;
                end
                
                for j = simNo
                    LassoBatch.ypred(:, i, j) = YC{j};
                    LassoBatch.RMSE(i, j)     = RC{j}.int;
                    LassoBatch.MAE(i, j)      = RC{j}.MAE;
                    LassoBatch.R2(i, j)       = R2C{j}.int;
                    LassoBatch.lambda(i, j)   = LC{j};
                    LassoBatch.vs{i, j}       = VC{j};
                    LassoBatch.Beta{i, j}     = BB{j};
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    LassoBatch.maxAE(i, j)    = sortedErr(end);
                    LassoBatch.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    LassoBatch.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));

                end
                clear PC YC RC R2C LC VC TC
                save(fname, 'LassoBatch', '-append')
                display(['Lag = ', num2str(pArray(i)-1), ' | Lasso RMSE: ', num2str(mean(LassoBatch.RMSE(i, :), 2))])
          
            end
        case 'adaptive'
            %%
            cd 'C:\Users\ChE\Google Drive\Local Methods'
            % JITL (w = 0) %
            aParam.tuneSW = '2.1c'; aParam.flagCV = 'Adaptive lastblock';
            aParam.sRatio = .35; flagA = 'SWCV'; aParam.timeFlag = true;
            % CheckSigma & MW-D
            aParam.cols = [2:5, 7, 8, 10, 14, 15, 17]+1;
            aParam.stdPEflag = false; aParam.alphaFlag = 1; aParam.lambda = 0.3;
            aParam.Inc = NaN; aParam.IncFlag = true; aParam.minvarFlag = true;
            aParam.sLim = 20; aParam.lowerLim = 20;
            % Others
            aParam.lowLimRange = aParam.lowerLim; aParam.sLimRange = aParam.sLim;
            aParam.Kc = 20; aParam.KcRange = aParam.Kc; aParam.tauI = 20; aParam.tauIRange = aParam.tauI;
            % JITL
            aParam.flagNN = 'basicNN-w1'; aParam.maxPast = 2; aParam.NS = 30;
            for i = 2
                aParam.WSize = WRange(i); aParam.WRange = aParam.WSize;
                parfor j = simNo
                     rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, ~, ~, ~, ~, tt] = adaptiveSBPrediction([trainSet' Xtrain(trainSet, pSel)],...
                        ytrain(trainSet), [(301:700)' Xtest(:, pSel)], ytest, aParam, flagA);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    TC{j} = tt;
                    
                end
                
                for j = simNo
                    JITL.ypred(:, i, j) = YC{j};
                    JITL.RMSE(i, j)     = RC{j}.int;
                    JITL.R2(i, j)       = R2C{j}.int;
                    JITL.ETime(i, j)    = mean(TC{j}.diagnose.timeCounterer);
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    JITL.MAE(i, j)      = mean(sortedErr);
                    JITL.maxAE(i, j)    = sortedErr(end);
                    JITL.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    JITL.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                 clear PC YC RC R2C LC VC TC
                save(fname, 'JITL', '-append')
                display(['W = ', num2str(WRange(i)), ' | JITL RMSE : ', num2str(mean(JITL.RMSE(i, :), 2))])
            end
            
            % JITLMW %
            aParam.tuneSW = '2.1c'; aParam.flagCV = 'Adaptive lastblock';
            aParam.sRatio = .35; flagA = 'SWCV'; aParam.timeFlag = true;
            % CheckSigma & MW-D
            aParam.cols = [];
            aParam.stdPEflag = false; aParam.alphaFlag = 1; aParam.lambda = 0.3;
            aParam.Inc = NaN; aParam.IncFlag = true; aParam.minvarFlag = false;
            aParam.sLim = Inf; aParam.lowerLim = Inf;
            % Others
            aParam.lowLimRange = aParam.lowerLim; aParam.sLimRange = aParam.sLim;
            aParam.Kc = 20; aParam.KcRange = aParam.Kc; aParam.tauI = 20; aParam.tauIRange = aParam.tauI;
            % JITL
            aParam.flagNN = 'basicNN'; aParam.maxPast = 2; aParam.NS = 30;
            for i = 2
                aParam.WSize = WRange(i); aParam.WRange = aParam.WSize;
                parfor j = simNo
                     rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, ~, ~, ~, ~, tt] = adaptiveSBPrediction([trainSet' Xtrain(trainSet, pSel)],...
                        ytrain(trainSet), [(301:700)' Xtest(:, pSel)], ytest, aParam, flagA);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    TC{j} = tt;
                    
                end
                
                for j = simNo
                    JITLMW.ypred(:, i, j) = YC{j};
                    JITLMW.RMSE(i, j)     = RC{j}.int;
                    JITLMW.R2(i, j)       = R2C{j}.int;
                    JITLMW.ETime(i, j)    = mean(TC{j}.diagnose.timeCounterer);
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    JITLMW.MAE(i, j)      = mean(sortedErr);
                    JITLMW.maxAE(i, j)    = sortedErr(end);
                    JITLMW.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    JITLMW.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                 clear PC YC RC R2C LC VC TC
                save(fname, 'JITLMW', '-append')
                display(['W = ', num2str(WRange(i)), ' | JITL + MW RMSE : ', num2str(mean(JITLMW.RMSE(i, :), 2))])
            end
            
            % JITLMW-D %
            aParam.tuneSW = '2.1c'; aParam.flagCV = 'Adaptive lastblock';
            aParam.sRatio = .35; flagA = 'SWCV';
            % CheckSigma & MW-D
            aParam.cols = [2:5, 7, 8, 10, 14, 15, 17]+1;
            aParam.timeFlag = true;
            aParam.stdPEflag = false; aParam.alphaFlag = 1; aParam.lambda = 0.3;
            aParam.Inc = NaN; aParam.IncFlag = true; aParam.minvarFlag = false;
            aParam.sLim = Inf; aParam.lowerLim = 20;
            % Others
            aParam.lowLimRange = aParam.lowerLim; aParam.sLimRange = aParam.sLim;
            aParam.Kc = 20; aParam.KcRange = aParam.Kc; aParam.tauI = 20; aParam.tauIRange = aParam.tauI;
            % JITL
            aParam.flagNN = 'basicNN'; aParam.maxPast = 2; aParam.NS = 30;
            for i = 2
                aParam.WSize = WRange(i); aParam.WRange = aParam.WSize;
                parfor j = simNo
                     rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, ~, ~, ~, ~, tt] = adaptiveSBPrediction([trainSet' Xtrain(trainSet, pSel)],...
                        ytrain(trainSet), [(301:700)' Xtest(:, pSel)], ytest, aParam, flagA);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    TC{j} = tt;
                    
                end
                
                for j = simNo
                    JITLMWD.ypred(:, i, j) = YC{j};
                    JITLMWD.RMSE(i, j)     = RC{j}.int;
                    JITLMWD.R2(i, j)       = R2C{j}.int;
                    JITLMWD.ETime(i, j)    = mean(TC{j}.diagnose.timeCounterer);
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    JITLMWD.MAE(i, j)      = mean(sortedErr);
                    JITLMWD.maxAE(i, j)    = sortedErr(end);
                    JITLMWD.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    JITLMWD.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                 clear PC YC RC R2C LC VC TC
                save(fname, 'JITLMWD', '-append')
                display(['W = ', num2str(WRange(i)), ' | JITL + MW-D RMSE : ', num2str(mean(JITLMWD.RMSE(i, :), 2))])
            end
            
        case 'adaptive-ultimate'
            %%
            cd 'C:\Users\ChE\Google Drive\Local Methods'
            % MWAdpJITL %
            aParam.tuneSW = '2.1c'; aParam.flagCV = 'Adaptive lastblock';
            aParam.sRatio = .35; flagA = 'SWCV'; aParam.timeFlag = true;
            % CheckSigma & MW-D
            aParam.cols = [2:5, 7, 8, 10, 14, 15, 17]+1;
            aParam.stdPEflag = false; aParam.alphaFlag = 1; aParam.lambda = 0.3;
            aParam.Inc = NaN; aParam.IncFlag = true; aParam.minvarFlag = true;
            aParam.sLim = 20; aParam.lowerLim = 20;
            % Others
            aParam.lowLimRange = aParam.lowerLim; aParam.sLimRange = aParam.sLim;
            aParam.Kc = 20; aParam.KcRange = aParam.Kc; aParam.tauI = 20; aParam.tauIRange = aParam.tauI;
            % JITL
            aParam.flagNN = 'basicNN'; aParam.maxPast = 2; aParam.NS = 30;
            for i = 1:length(WRange)
                aParam.WSize = WRange(i); aParam.WRange = aParam.WSize;
                parfor j = simNo
                    rng(SD(j))
                    Xtrain = XtrainC{j};
                    Xtest  = XtestC{j};
                    ytrain = ytrainC{j};
                    ytest  = ytestC{j};
                    [ypred, RMSEpred, R2pred, ~, ~, ~, ~, tt] = adaptiveSBPrediction([trainSet' Xtrain(trainSet, pSel)],...
                        ytrain(trainSet), [(301:700)' Xtest(:, pSel)], ytest, aParam, flagA);
                    YC{j} = ypred.int;
                    RC{j} = RMSEpred;
                    R2C{j} = R2pred;
                    TC{j} = tt;
                    
                end
                
                for j = simNo
                    MWAdpJITL.ypred(:, i, j) = YC{j};
                    MWAdpJITL.RMSE(i, j)     = RC{j}.int;
                    MWAdpJITL.R2(i, j)       = R2C{j}.int;
                    MWAdpJITL.ETime(i, j)    = mean(TC{j}.diagnose.timeCounterer);
                    sortedErr = sort(abs(ytestC{j} - YC{j}));
                    MWAdpJITL.MAE(i, j)      = mean(sortedErr);
                    MWAdpJITL.maxAE(i, j)    = sortedErr(end);
                    MWAdpJITL.RMSE99(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.99)).^2));
                    MWAdpJITL.RMSE90(i, j)   = sqrt(mean(sortedErr(1:round(length(sortedErr)*.90)).^2));
                end
                clear PC YC RC R2C LC VC TC
                save(fname, 'MWAdpJITL', '-append')
                display(['W = ', num2str(WRange(i)), ' | MWAdpJITL RMSE : ', num2str(mean(MWAdpJITL.RMSE(i, :), 2))])
            end
    end
end

