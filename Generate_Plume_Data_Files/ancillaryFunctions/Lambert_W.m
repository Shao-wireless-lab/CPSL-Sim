function w = Lambert_W(branch, x)
% Lambert_W Lambert's W function.
%    W = lambertw(Z) solves W*exp(W) = Z.
%    W = lambertw(K,Z) is the K-th branch of this multi-valued function.
 
%    References:
%    [1] Robert M. Corless, G. H. Gonnet, D. E. G. Hare,
%    D. J. Jeffrey, and D. E. Knuth, "On the Lambert W Function",
%    Advances in Computational Mathematics, volume 5, 1996, pp. 329-359.
 
%    [2] Corless, Jeffrey, Knuth, "A Sequence of Series
%    for The Lambert W Function", ISSAC '97
%    Copyright Lateef Adewale Kareem 2022.
if nargin < 2
    x = branch;  branch = 0;
end
% Effective starting guess
v = inf*ones(size(x));
if numel(branch) == 1
    if branch == 0
       w = ones(size(x));  % Start above -1
    else  
       w = -2*ones(size(x));  % Start below -1
    end
    if(branch == 0 || branch  == -1)
        w = Haley(w, v, x, 0);
    else
        w = Haley(w, v, x, branch);
    end
else
    w = ones(size(x));
    w(branch ~= 0) = -2;
    indx = branch == 0 | branch == -1;
    w(indx) = Haley(w(indx), v(indx), x(indx), 0);
    indx = ~(branch == 0 & branch ==-1);
    w(indx) = Haley(w(indx), v(indx), x(indx), branch(indx));
end
w(x==0)=0;
w(x==-1/exp(1))=-1;
end
function w = Haley(w, v, x, branch)
    % Haley's method
    while any(abs(w - v)./abs(w) > 1.e-15)
       v = w;
       f = w + log(w) - log(x) - 2*pi*1i*branch;
       fp = 1 + 1./w;
       fpp = -1./(w.*w);
       % Iterate to make this quantity zero
       w = w - f ./ (fp - f .* fpp ./ (2 * fp));
    end
end
