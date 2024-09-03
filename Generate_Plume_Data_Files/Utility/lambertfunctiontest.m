clear
clc

% define discretized space
xb = [0,100];
yb = [0,100];
n = 100;
m = 100;
x = linspace(xb(1),xb(2),n)';
y = linspace(yb(1),yb(2),m)';
[xx,yy] = meshgrid(x,y);
X = [xx(:),yy(:)];

% define the source and env
% xs = [rand*(xb(2)-xb(1))+xb(1),...
%     rand*(yb(2)-yb(1)) + yb(1)];
xs = [20,50];
q0 = 1; tau0 = 0.1; U = 1;
y0 = xs(2); x0 = xs(1);
p = [q0,y0,tau0];

% define the sensor and path
nz = 10;
eta_m = (1e-1)^2;
eta_b = (1e-2)^2;
dwd = 60;
z0 = xs(1)+dwd;
z = [z0*ones(nz,1),linspace(40,60,nz)'];

% create model function
M = @(x,p) (p(1)./(2*U*pi*x(:,1).*p(3))) .* exp(-0.5*((x(:,2)-p(2))./(x(:,1)*p(3))).^2);

% take ground truth measurement
temp = [X(:,1)-x0,X(:,2)];
% Y(temp<0) = 0;
% Y(temp>=0) = M(X(temp>=0,:),p);
Y = M(temp,p);
Y(Y<0) = 0;

% take mobile sensor measurement
temp = [z(:,1)-x0,z(:,2)];
Yz = (1+eta_m*randn(nz,1)).*M(temp,p) + eta_b*randn(nz,1);
Yz(Yz<0) = 0;

% estimate Diffusion coef used to relate the ADE solution to GPM
cf = 1; % correction factor
K = cf*p(3)^2*(dwd)*U/2;

% define lambertw distance levelset function handle
d = @(xi,Ci,x0,q0) (2*K/U)*lambertw((U.*q0./(4*pi.*K^2.*Ci))*exp((U./(2.*K)).*(xi-x0)));
% testnum_id = [1,round(nz)/2,nz];
for j=1:nz
% testnum = testnum_id(j);
testnum = j;
n0 = 1000;
yi = z(testnum,2);
xi = z(testnum,1);
Ci = Yz(testnum);
x0 = linspace(x(1),x(end),n0)';
x0 = [x0;x0];
tempy = zeros(2*n0,1);
for i=1:n0
    tempy(i) = yi + (sqrt(d(xi,Ci,x0(i),q0)^2-(xi-x0(i)).^2));
    tempy(n0+i) = yi - (sqrt(d(xi,Ci,x0(i),q0)^2-(xi-x0(i)).^2));    
end
tempy(abs(imag(tempy))>0) = nan;
y0i(:,j) = tempy;
% temp = (abs(imag(y0i))>0);
% y0i(temp)=[];
% x0(temp)=[];
end

% optimize over x0 and q0
temp = [y0i(1:n0,:)';y0i(n0+1:end,:)'];
J = zeros(2*nz,2*nz,n0);
for k=1:n0
    for i=1:nz
        if 1%~isnan(temp(i,k)) 
            for j=1:nz
                if  ~isnan(temp(j,k))
                    a(1) = (temp(i,k)-temp(j,k))^2;
                    a(2) = (temp(i,k)-temp(j+nz,k))^2;
                    a(3) = (temp(i+nz,k)-temp(j,k))^2;
                    a(4) = (temp(i+nz,k)-temp(j+nz,k))^2;
                    J(i,j,k) = min(a,[],'omitnan');
                else
                    J(i,j,k) = 1e3;
                end

            end
        end
    end
    Em(k) = sum(J(:,:,k),'all','omitnan');
end
J = Em;
k = find(J==min(J));

% plot 
figure(1)
YY = reshape(Y,[n,m]);
surf(xx,yy,YY)
shading interp
alpha(0.2)
view(2)
hold on
plot(xs(1),xs(2),'r*')
plot3(z(:,1),z(:,2),Yz,'bs')
for i=1:size(y0i,2)
    plot(x0,real(y0i(:,i)),'b.')
end
plot(x0(k),mean(temp(:,k)),'gd')
hold off
grid on

axis([xs(1)-10,z0+10,xs(2)-30,xs(2)+30])
figure(2)
plot(x0(1:n0),J,'k-')
ylim([0,inf])
xlim([x(1),x(end)])
grid on
