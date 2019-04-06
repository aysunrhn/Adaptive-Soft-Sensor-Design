function [ypred, mu, hyperparameter] = SBPrediction (Xtrain, ytrain, Xtest, ytest, likelihood, tol)
Ntrain = length(ytrain);
Ntest  = length(ytest);
[Xsc, mx, sx] = auto(Xtrain);
[ysc, my, sy] = auto(ytrain); ysc(isnan(ysc)) = 1e-10;
Xtestsc = (Xtest - repmat(mx, [Ntest, 1]))./repmat(sx, [Ntest, 1]);
vars = setdiff(1:size(Xtrain, 2), find(~range(Xtrain)));

CONTROLS = SB2_ControlSettings;
CONTROLS.ZeroFactor = tol(1);
CONTROLS.MinDeltaLogAlpha = tol(2);
CONTROLS.MinDeltaLogBeta = tol(3);

[parameter, hyperparameter, ~] = SparseBayes(likelihood, Xsc(:, vars), ysc, CONTROLS);
vSelected = vars(parameter.Relevant);
mu = zeros(size(Xtrain, 2), 1);
mu(vSelected) = parameter.Value;
ypred = Xtestsc(:, vSelected)*parameter.Value.*repmat(sy, [Ntest, 1]) + repmat(my, [Ntest, 1]);

sigma2  = 1/hyperparameter.beta;
invSS   = hyperparameter.beta*Xsc(:, vSelected)'*Xsc(:, vSelected) + diag(hyperparameter.Alpha);
SS      = inv(invSS);
Psigma2 = (sigma2 + diag(Xtestsc(:, vSelected)*SS*Xtestsc(:, vSelected)'))*var(ytrain);
hyperparameter.Psigma2     = Psigma2;
hyperparameter.modelSigma2 = sigma2*var(ytrain);
