function [Xtrain, ytrain, flag, WNew, trainIdx, dVec, WVec, dNew] = mahalDistCheck(xq, X, y, WSize, cols)

incP  = 1.2;
alpha = .99;
flag  = false;
cols  = 1:min(length(cols), size(X, 2));
p  = length(cols);
LB99  = p*(WSize-1)*(WSize+1)/WSize/(WSize - p)*finv(alpha, p, WSize - p);
LB995 = p*(WSize-1)*(WSize+1)/WSize/(WSize - p)*finv(.995, p, WSize - p);
N  = length(y); Xtrain = X((N - WSize + 1):N, :);
d0 = mdist(xq(:, cols), Xtrain(:, cols));
dVec = d0;
WVec = WSize;
if d0 > LB99
    flag = true;
    while dVec(end) > LB99
       WVec = [WVec; round(WVec(end)*incP)];
       if WVec(end) > N
          take2 = find(dVec < LB995, 1);
          if ~isempty(take2)
             idx = take2;
          else
              [~, idx] = min(dVec);
          end
          WNew = WVec(idx);
          dNew = dVec(idx);
          trainIdx  = (N - WNew + 1):N;
          Xtrain    = X(trainIdx, :); 
          ytrain    = y(trainIdx);
          WVec(end) = [];
          return
       end
       nIdx = (N - WVec(end) + 1):N; Xtrain = X(nIdx, :); ytrain = y(nIdx);       
       LB99  = p*(WVec(end)-1)*(WVec(end)+1)/WVec(end)/(WVec(end)- p)*finv(alpha, ...
               p, WVec(end) - p);
       LB995 = [LB995; p*(WVec(end)-1)*(WVec(end)+1)/WVec(end)/(WVec(end) - ...
               p)*finv(.995, p, WVec(end) - p)];
       dVec = [dVec; mdist(xq(:, cols), Xtrain(:, cols))];  
    end
end
WNew = WVec(end);
dNew = dVec(end);
trainIdx = (N - WNew + 1):N;
Xtrain   = X(trainIdx, :);
ytrain   = y(trainIdx);
return
end
function d = mdist(Y,X)
nx = size(X, 1); ny = size(Y, 1);
m = mean(X, 1);
M = m(ones(ny, 1), :);
C = X - m(ones(nx, 1), :);
[Q, R] = qr(C,0);

ri = R'\(Y - M)';
d = sum(ri.*ri,1)'*(nx - 1);
end