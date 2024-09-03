%% Run localization experiment for Lambert W 
% by Derek Hollenbeck
% 
% This script applies the LambertW based localization algorithm on
% preprocessed measurement data (in the gdat format). 
%
% References
% [1] van Ulden, A.P. Simple Estimates For Vertical Diffusion From Sources
% Near The Ground. Atmospheric Environment. 1978
% [2] Matthes, J., Groll, L., and Keller, H.B. Source Localization by
% Spatially Distributed Electronic Noses for Advection and Diffusion. IEEE
% Transactions on Signal Processing. 2005.
% [3] Langenberg, S., Carstens, T., Hupperich, D., Schweighoefer, S., and
% Schurath, U. Technical Note: Determination of binary gas-phase diffusion
% coefficients of unstable and adsorbing stmospheric trace gases at low
% temperature - arrest flow and twin tube method. Atmosphere. 2020.
%

%-------------------------------------------------------
% -- Psudocode 
% 0a. Load data
% 0b. Cut data (to remove `bad' data) -- in the sense of localization
% 0c. Set optimization grid size and options/params
% 1. Rotate data in local sense, relative to wind 
% 2. Apply initial guess on dispersion scale factor -- using MOST
% 3. Apply Lambert optimization (least squares sense) -- get source rate
% and localization estimate.
%
% -- Assumptions
% 1. Homogenous wind field
% 2. Averaged measurements
% 3. 
%-----------------------------------------------

% -- Add paths and clear workspace
clear, clc
addpath ancillaryFunctions\
addpath data\
addpath figures\

% -- Settings
fn = 'C:\Users\dhollenbeck\Documents\MATLAB\data\data_2023_04_27';
outfolder = 'C:\Users\dhollenbeck\Documents\MATLAB\figures\';
fltnum = 7;
load([fn,'_f',num2str(fltnum),'.mat'])
methodType = 'none';
format long g

% -- Plot Settings
PLOTDATA = 0;
windsize = 0.0003;
ubarscale = 0.0001;




% -- Load Data
xs = [srcdata(1:2),0];                  % Source Location, no altitude measurement
q = srcdata(3);                         % Source Rate, kg/s
z = fltdata(:,7:9);                     % Mobile Measurement position (lat,lon,alt)
y = fltdata(:,2);                       % Concentration Measurement (ratio)
yn = (y-min(y))/(max(y)-min(y));        % Normalized Concentration Measurement (for plotting mainly)
U = fltdata(:,11);                      % Wind Speed, m/s
psi = fltdata(:,12);                    % Wind Direction, degrees
u = [U.*sind(psi), U.*cosd(psi),0*U];   % Wind Vector, m/s

% -- Cut Data

% -- Set Optimization Grid


% -- Get Rotated Values
ubar = mean(u);                         % Average Wind Vector, m/s    
psibar = getWD(ubar);                   % Average Wind Direction, degrees
xl = z - xs;                            % Local Coord, relative to source estimate    
varphi = psibar + 90;                   % Angle to rotate
xr = rot(-varphi,xl);                   % Rotation Coord, for computing GPM, xr1 >= 0
uprime = rot(-(psibar+90),ubar);

% -- Apply Initial Guess on Scale Factors
Q_0 = 0.001;
mu_20 = 0;
mu_30 = 0;
tau_20 = 0.01;
tau_30 = 0.01;
theta_0 = [Q_0,mu_20,mu_30,tau_20,tau_30];

% -- Apply Least Squares Optimization w/ LambertW 
W = @() Lambert_W()
%%%%%%%%%%%%%%%%%%%%%%%--MAIN--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PLOTDATA == 1
% -- Plot Local Values
figure(1)
scatter3(xr(:,1),xr(:,2),xr(:,3),100,yn,'Marker','.')
hold on
scatter3(0,0,0,'r*')
plot(ubarscale*[0,uprime(1)],ubarscale*[0,uprime(2)],'b')
hold off
view(2)
axis([-windsize,windsize,-windsize,windsize])
saveas(gcf,[outfolder,'localpath_f',num2str(fltnum)],'png')

% -- Plot Global Values
figure(2)
scatter3(z(:,1),z(:,2),z(:,3),100,yn,'Marker','.')
hold on
scatter3(xs(1),xs(2),xs(3),'r*')
plot([xs(1),xs(1)+ubarscale*ubar(1)],[xs(2),xs(2)+ubarscale*ubar(2)],'b')
hold off
view(2)
axis([xs(1)-windsize,xs(1)+windsize,xs(2)-windsize,xs(2)+windsize])
saveas(gcf,[outfolder,'globalpath_f',num2str(fltnum)],'png')

% -- Plot Wind Rose
[figure_handle,count,speeds,directions,Table,Others] = WindRose(...
    psi,U,...
    'anglenorth',0,...
    'angleeast',90,...
    'labels',{'N (0º)','S (180º)','E (90º)','W (270º)'},...
    'freqlabelangle',45);
saveas(figure_handle,[outfolder,'windrose_f',num2str(fltnum)],'png')
% clf(figure_handle)
end

% -- Functions
function varphi = getWD(u)
varphi = atan2d(u(1),u(2));
end

function phi = getCosAng(u,v)
phi = acosd(dot(u,v)/(norm(u)*norm(v)));
end

function z = localframe(z,xs,u)
x = z-xs;
varphi = getWD(u);
uprime = rot(varphi,u);
phi = getCosAng(uprime,[0,1,0]);
z = rot(-(phi),x);
end

function z = rot(phi,z)
for i=1:size(z,1)
    z(i,:) = z(i,:)*[cosd(phi),-sind(phi),0;sind(phi),cosd(phi),0;0,0,1];
end
end