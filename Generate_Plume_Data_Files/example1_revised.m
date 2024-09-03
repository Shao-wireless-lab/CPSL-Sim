%% Example 1
% This script runs a MOABS/DT simulation where the plume location
% is estimated via the particle filter. 

% -- INITIALIZE SETTINGS AND ADD FOLDER PATHS
clc, clear, close all  
addpath Utility
addpath Terrain
addpath Classes
example_1_settings_revised
PLOTRESULTS = 1;
TERRAIN_ON = 0;

% -- DATA STORED
fn = 'telem_testing.csv';

% -- INITIALIZE SIMULATION OBJECTS
ff = flowfieldClass(ff_params);
plume = filamentClass(plume_params);
plan = plannerClass(planner_params);

% -- INITIALIZE VEHICLE / PLANNER OBJECTS
veh = vehicleClass(vehicle_params);
swarmnum = 10; temp = ones(swarmnum,1);
tempx = linspace(-plan.w/2,plan.w/2,swarmnum)';
tempz = 2*ones(swarmnum,1);
tempy = 0*tempx;
p = plan.getPlaneGlobal([tempx,tempy,tempz],plan.r,plan.phi);
clear tempx tempy tempz
for i=1:swarmnum
    veh.p = p(i,:);    
    veh = veh.setWaypoints(veh.p);
    plan = plan.addVehicle(veh);
end

% -- initialize environment
env = environmentClass();
env = env.addFlowfield(ff); clear ff;
env = env.addPlume(plume); clear plume;
env = env.addPlanner(plan); clear plan;
%if TERRAIN_ON
%    env = env.addTerrain();
%    env.terrain = env.terrain.clip2window(...
%        [env.ff.xmin,env.ff.xmax,env.ff.ymin,env.ff.ymax]);
%end
env.az = 30;
env.el = 30;

% -- main loop settings 
n = simparams.N*env.Fs;
mod_data = round(env.Fs/env.Fs_data);

dataout = []; 
fillament_locs = cell(1,n); 
% 
DATA_Height_2 = zeros(100*n,100*4); 
DATA_Height_5 = zeros(100*n,100*4); 

F1 = []; 


X = [0 200 200 0]; 
Y = [0 0 200 200]; 
Z = [2,5;2,5;2,5;2,5]; 

% -- MAIN LOOP 
fig1 = figure(1); 
tic 
%try     
for i=1:6000
        % -- adjust CVT plane position
        %if i == 200
            %env.planner.rd = [40,50,5];
            %env.planner.phid = 90;
        %end

        % -- step environment
        env = env.step();

        %DATA{i} = env.planner.dataout2; 
        DATA_Height_2(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout2)]; 
        DATA_Height_5(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout5)]; 
        fillament_locs{i} = env.fil_locs; 
        disp(i)
        % -- record variables
        % if mod(i,mod_data)
        %     dataout = [dataout;...
        %         env.planner.k*env.planner.dt,...
        %         env.planner.Mdot,...
        %         env.planner.P_hat...
        %         ];
        %     %fprintf(fileID,formatSpec,dataout);
        %     %writematrix(dataout,fn,'WriteMode','append')
        % end        
        % if i == n
        %     t_sim = toc;
        %     writematrix(dataout,fn)
        % end

        % -- plot results
        % if mod(i,env.Fs)==0 && PLOTRESULTS
        %     subplot(1,3,1)
        % 
        %     env.plotff(1)
        %     %env.plotplume(1)
        %     hold on; 
        %     scatter3(fillament_locs{i}(:,1),fillament_locs{i}(:,2),fillament_locs{i}(:,3),1000.*fillament_locs{i}(:,4),'r.')
        %     %env.plotplanner(1)
        %     %F{i} = env.getFrame(); % -- for saving to gif 
        %     hold on; 
        %     fill3(X,Y,Z(:,1),[0.7 0.7 0.7],'FaceAlpha',0.3)
        %     hold on; 
        %     fill3(X,Y,Z(:,2),[0.7 0.7 0.7],'FaceAlpha',0.3)
        %     grid on; 
        %     xlabel('X (m)')
        %     ylabel('Y (m)')
        %     zlabel('Z (m)')
        %     set(gca,'FontName','Times','FontSize',12,'FontWeight','bold')
        %     view(-37.5,30)
        %     xlim([0 200])
        %     ylim([0 200])
        %     zlim([0 25])
        % 
        %     subplot(1,3,2)
        %     env.plotff(1)
        %     %env.plotplume(1)
        %     hold on; 
        %     scatter3(fillament_locs{i}(:,1),fillament_locs{i}(:,2),fillament_locs{i}(:,3),1000.*fillament_locs{i}(:,4),'r.')
        %     hold on; 
        %     fill3(X,Y,Z(:,1),[0.7 0.7 0.7],'FaceAlpha',0.3)
        %     hold on; 
        %     fill3(X,Y,Z(:,2),[0.7 0.7 0.7],'FaceAlpha',0.3)
        %     grid on; 
        %     xlabel('$X$ (m)','Interpreter','latex')
        %     ylabel('$Y$ (m)','Interpreter','latex')
        %     zlabel('$Z$ (m)','Interpreter','latex')
        %     set(gca,'FontName','Times','FontSize',12,'FontWeight','bold')
        %     view(0,0)
        %     xlim([0 200])
        %     ylim([0 200])
        %     zlim([0 25])
        % 
        %     subplot(1,3,3)
        % 
        %     % close all; 
        %     % figure(2)
        %     % i = 5900; 
        %     env.plotff(1)
        %     %env.plotplume(1)
        %     hold on; 
        %     scatter3(fillament_locs{i}(:,1),fillament_locs{i}(:,2),fillament_locs{i}(:,3),1000.*fillament_locs{i}(:,4),'r.')
        %     hold on; 
        %     %fill3(X,Y,Z(:,1),'r','FaceAlpha',0.3)
        %     hold on; 
        %     %fill3(X,Y,Z(:,2),'r','FaceAlpha',0.3)
        %     grid on; 
        %     xlabel('$X$ (m)','Interpreter','latex')
        %     ylabel('$Y$ (m)','Interpreter','latex')
        %     %zlabel('$Z$ (m)','Interpreter','latex')
        %     set(gca,'FontName','Times','FontSize',12,'FontWeight','bold')
        %     view(0,90)
        %     xlim([0 200])
        %     ylim([0 200])
        %     zlim([0 25])
        % 
        %     shg
        %     F1 = [F1; getframe(fig1)];
        %     hold off; 
        %     clf
        % end
        
    
end
%catch
F1 = [F1; getframe(gcf)];

% create the video writer with 1 fps
writerObj = VideoWriter('Plume-Animation-With-Meander.mp4');
writerObj.FrameRate = 10;
% set the seconds per image
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(F1)
% convert the image to a frame
frame = F1(i) ;    
writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);

%%
close all; 
figure(4)
ind = 2560;
for ind = 3200
    scatter3(fillament_locs{ind}(:,1),fillament_locs{ind}(:,2),fillament_locs{ind}(:,3),1000.*fillament_locs{ind}(:,4),'r.')
    view(0,90)
    pause(0.1)
end
%env.plotff(1)
hold on; 
scatter3(80,60,5,150,'g.')
view(0,90)
xlabel('$X$ (m)','Interpreter','latex')
ylabel('$Y$ (m)','Interpreter','latex')
zlabel('$Z$ (m)','Interpreter','latex')
set(gca,'FontName','Times','FontSize',12,'FontWeight','bold')


%end
% t_end = toc;
% t_rate = simparams.N/t_sim;
% t_rateToEnd = simparams.N/t_end;




% %% Process data
% datain = importdata('telem_testing.csv',',');
% t = datain(:,1);
% mdot = datain(:,2);
% thet = datain(:,3:7);
% 
% % figure(2)
% % plot(t,mdot, t, movmean(mdot,1500))
% % 
% % ylim([0,30])
