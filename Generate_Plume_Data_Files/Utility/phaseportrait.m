clear
clc

h = 20; w = 20;
dh = 1; dw = 1;
x0 = [0,h/2];
xc = [0,0];
x1 = -w/2:dw:w/2 ;
x2 = -h/2:dh:h/2 ;
[xx1,xx2] = meshgrid(x1+ x0(1),x2+ x0(2));
n = length(x1);
X = [xx1(:),xx2(:)];
vmax = 1;
lam = 0.05;
gam = -0.5;
bet = 0;
% A = eye(2);
a11 = bet;
a12 = 1;
a21 = gam;
a22 = bet;
A = [a11,a12;a21,a22];
% A = -A;
% f = @(x) [-1/(2*pi) * (x(:,1)-0) ./ (x(:,1) - 0).^2, -1/(2*pi) * (x(:,2)-10) ./ (x(:,2) - 2).^2];
% B = [0,1;1,0];
% u = [1;1];

% compute the derivative
xdot = ((lam*A/norm(A))*(X-xc-x0)')';
% xdot = (A*X'+B*u)';
% xdot = real(f(X));
% xdot(xdot>vmax) = vmax*sign(xdot(xdot>vmax));
x1dot = reshape(xdot(:,1),[n,n]);
x2dot = reshape(xdot(:,2),[n,n]);
xx1p = xx1; %+ x0(1); 
xx2p = xx2; %+ x0(2);
surf(xx1p,xx2p,sqrt(x1dot.^2+x2dot.^2))
shading interp
view(2)
alpha(0.5)
caxis([0,vmax]);
hold on
quiver(xx1p,xx2p,x1dot,x2dot,2)
hold off
axis([x1(1)+x0(1),x1(end)+x0(1),...
    x2(1)+x0(2),x2(end)+x0(2)])