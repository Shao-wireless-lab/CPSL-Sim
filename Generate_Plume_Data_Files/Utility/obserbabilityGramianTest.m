addpath Utility\

load cproj_t.mat

ep = 1;
s = size(xx);
surf(xx,yy,Cproj_t_avg)
% shading interp
view(2)
alpha(0.7)

hold on
npaths = 3;
for j=1:npaths
nstep = 5;
x0 = [100,140];
x = zeros(nstep,2);
x(1,:) = x0;
ex = [1,0;0,1;-1,0;0,-1];
for i=2:nstep
    x(i,:) = x(i-1,:) + [round(rand()*2-1),round(rand()*2-1)];
end

plot(x(:,1),x(:,2),'k')
plot(x(end,1),x(end,2),'ro')
end
plot(x0(1),x0(2),'g*')
hold off
