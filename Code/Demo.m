%% MWAdpJITL DEMO %%
% See MWAdpJITL.m for detailed explanation on parameters

% Load Debutanizer column dataset (Fortuna, 2007)
% https://www.springer.com/gp/book/9781846284793
data   = load('debutanizer_data.txt');
Xtrain = data(1:1000, 1:end-1);
ytrain = data(1:1000, end);
Xtest  = data(1001:end, 1:end-1);
ytest  = data(1001:end, end);

%% 1. MWAdp
flagM = 'MWAdp'; % method name
paramList.tFlag  = false; % time indices are not used
paramList.WSize  = 30;    % WU = 30
paramList.WL     = 8;
paramList.WRange = []; % WU is preset, do not perform CV to optimize WU
paramList.mahal  = 1:size(Xtrain, 2); % use all predictors in CheckM
paramList.sLFlag = false; % use default value of sL
paramList.alpha  = 1; % 99.9% significance level 
paramList.lambda = 0.3; % use lambda = 0.3 in EWMA
[ypred, err] = MWAdpJITL(Xtrain, ytrain, Xtest, ytest, flagM, paramList);

%% 2. MWAdpJITL
flagM = 'MWAdpJITL'; % method name
paramList.tFlag  = false; % time indices are not used
paramList.WSize  = 30; % WU = 30
paramList.WL     = 8;
paramList.WRange = []; % WU is preset, do not perform CV to optimize WU
paramList.mahal  = 1:size(Xtrain, 2); % use all predictors in CheckM
paramList.sLFlag = false; % use default value of sL
paramList.alpha  = 1; % 99.9% significance level 
paramList.lambda = 0.3; % use lambda = 0.3 in EWMA
paramList.NNType = 'def'; % MWAdpJITL, no constraint on training set size
paramList.Wtilde = 20; % use Wtilde = 20, # of observations in JITL model
paramList.TSMax  = []; % empty because training set size is not fixed
paramList.mp     = 3; % # of validation points used in JITL subroutine
[ypred, err] = MWAdpJITL(Xtrain, ytrain, Xtest, ytest, flagM, paramList);

%% 3. MWAdpJITL (lim)
paramList.NNType = 'lim'; % MWAdpJITL (lim), contraint on training set size
[ypred, err] = MWAdpJITL(Xtrain, ytrain, Xtest, ytest, flagM, paramList);
