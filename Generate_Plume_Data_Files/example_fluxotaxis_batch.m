%% Fluxotaxis single run script
% This script runs a batch script of MOABS/DT simulations with three agents
% using the Fluxotaxis algorithm

% -- INITIALIZE SETTINGS AND ADD FOLDER PATHS
clc, clear, addpath Utility\ Terrain\ Classes\
numtrials_per_setting = 100;
G = [ones(1,numtrials_per_setting),3*ones(1,numtrials_per_setting),5*ones(1,numtrials_per_setting)];
numtrials = length(G);

% -- MAIN LOOP SETTINGS
example_fluxotaxis_batch_settings
n = simparams.N*20;
mod_data = 2;
dataout = zeros(n/mod_data,2*numtrials+1);
score = zeros(length(G),2);
PLOTRESULTS = 0;
TERRAIN_ON = 0;

% -- DATA STORED
fn = 'fluxotaxis_batch.csv'; 

% -- MAIN TRIAL LOOP
for trial = 1: numtrials

    % -- re-initialize settings for new trial
    disp(['trial ',num2str(trial)])
    example_fluxotaxis_batch_settings
    ff_params.abG = [0.005,0.02,G(trial)];

    % -- INITIALIZE SIMULATION OBJECTS
    ff = flowfieldClass(ff_params);
    plume = filamentClass(plume_params);
    plan = plannerClass(planner_params);

    % -- INITIALIZE VEHICLE / PLANNER OBJECTS
    veh = vehicleClass(vehicle_params);
    swarmnum = 3;
    p0 = [130,60,5];
    for i=1:swarmnum
        p(i,1) = p0(1)+5*cos((i-1)*2*pi/swarmnum);
        p(i,2) = p0(2)+5*sin((i-1)*2*pi/swarmnum);
    end
    p(:,3) = 5;
    waypoints = p(:,2) - 10;
    plan.vehicle.p = p(1,:);
    plan.vehicle.vehicleMode = 1;
    for i=2:swarmnum
        veh.p = p(i,:);
        veh.vehicleMode = 1;
        veh = veh.setWaypoints(veh.p);
        plan = plan.addVehicle(veh);
    end

    % -- initialize environment
    env = environmentClass();
    env = env.addFlowfield(ff); clear ff;
    env = env.addPlume(plume); clear plume;
    env = env.addPlanner(plan); clear plan;
    if TERRAIN_ON
        env = env.addTerrain();
        env.terrain = env.terrain.clip2window(...
            [env.ff.xmin,env.ff.xmax,env.ff.ymin,env.ff.ymax]);
    end
    % env.az = 30; env.el = 30;
    env.az = 0; env.el = 90;

    % -- main loop settings
    count = 0;

    % -- MAIN LOOP
    tic
    for i=1:n
        % -- adjust CVT plane position
        if i == env.Fs*140
            for k=1:swarmnum
                %env.planner.vehicle(k).setWaypoints(waypoints(k,:));
                env.planner.vehicle(k).vehicleMode = 3;
            end

        end

        % -- step environment
        env = env.step();

        % -- record variables
        if mod(i,mod_data) && exist('fn','var')
            for k = 1:1
                x = env.planner.Position(:,1:2);
                xmean = mean(x,1);
            end
            count = count + 1;
            dataout(count,1) = env.planner.k*env.planner.dt;
            dataout(count,trial*2:trial*2+1) = xmean;
            %fprintf(fileID,formatSpec,dataout);
            %writematrix(dataout,fn,'WriteMode','append')
        end
        if i == n
            t_sim = toc;
            if exist('fn','var')
                writematrix(dataout,fn)
            end
        end

        % -- plot results
        if mod(i,2*env.Fs)==0 && PLOTRESULTS
            figure(1)
            clf
            if ~isempty(env.terrain)
                env.plotterrain(0)
            end
            env.plotff(1)
            env.plotplume(1)
            env.plotplanner(1)
            %env = env.getFrame(); % -- for saving to gif
            shg
        end

    end
    t_end = toc;
    t_rate = simparams.N/t_sim;
    t_rateToEnd = simparams.N/t_end;
    score(trial,:) = [G(trial),norm(xmean-plume_params(1,1).p0(1:2))];
end
if exist('fn','var')
    writematrix(dataout,fn)
end


