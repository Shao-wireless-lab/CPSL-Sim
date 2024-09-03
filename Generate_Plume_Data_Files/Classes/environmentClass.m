%% Environment Class
% ----------------------
% by Derek Hollenbeck
%
% This class creates an environment object that acts as a container for
% running the MOABS/DT simulation
%
% EXAMPLE:
%   ...
% ----------------------

classdef environmentClass

    properties
        % -- DEFAULT ENVIRONMENT PARAMS
        xmin    = -50;
        xmax    = 150;
        ymin    = -50;
        ymax    = 150;
        zmin    = 0;
        zmax    = 20;
        fil_locs = []; 
        PLOTAZM = 45;
        PLOTELV = 40;
        Fs      = 20;
        dt      = [];
        ff      = flowfieldClass.empty; 
        plume   = filamentClass.empty; 
        planner = plannerClass.empty;
        terrain = terrainClass.empty;
        az      = 0;
        el      = 90;
        frames  = 0;
        fn      = 'test.gif';
        Fs_data = 10;
        
        %ColorSettings = {'r.','g.','b.'};
		ColorSettings = {'r.','b.'};
    end

    properties (Access = private)
        % properties exclusive to environment class
        currentTime = 0;
        NumFF = 1;
        NumPlumes = [1,1];
        NumPlanners = 0;      
    end

    methods

        % -- Constructor
        function obj = environmentClass(Sim_Params)
            
            % -- Check for initializiation parameters
            if nargin == 1
                % -- ADDITIONAL PARAMS
                %ff_params = [];

                % -- ATTACH Flowfield CLASS OBJECTS:
                if isfield(Sim_Params,'ff_params')
                    obj.NumFF = length(Sim_Params.ff_params);
                    for i = 1:obj.NumFF
                        obj.ff(i)      = flowfieldClass(Sim_Params.ff_params(i));     % Create flow field
                        fprintf('Flow Field(s) Initialized \n')
                    end
                else
                    obj.ff      = flowfieldClass();     % Create flow field 
                    obj.NumFF = 1;
                    fprintf('Initialized Single Flow Field with Default Settings \n')
                end
                
                % -- ATTACH Filament CLASS OBJECTS:
                if isfield(Sim_Params,'plume_params')
                    for i = 1:size(Sim_Params.plume_params,1)
                        for j = 1:size(Sim_Params.plume_params,2)
                            obj.plume(i,j)       = filamentClass(Sim_Params.plume_params(i,j));      % Create plume (In progress)
                        end
                        fprintf('Plume(s) Initialized \n')
                    end
                    obj.NumPlumes = size(Sim_Params.plume_params);
                else
                    obj.plume       = filamentClass();      % Create plume (In progress)
                    obj.NumPlumes = 1;
                    fprintf('Initialized Single Plume with Default Settings \n')
                end

                % -- ATTACH Planner CLASS OBJECTS:
                if isfield(Sim_Params,'planner_params')
                    obj.NumPlanners = length(Sim_Params.planner_params);
                    for i = 1:obj.NumPlanners
                        obj.planner(i) = plannerClass(Sim_Params.planner_params(i));      % Create plume (In progress)
                    end
                    fprintf('Planner(s) Initialized \n')
                else
                    obj.planner       = plannerClass();
                    fprintf('Initialized Single Planner with Default Settings \n')
                end

                % -- OTHER CONFIGURATIONS
                %obj.Fs = obj.ff.Fs;
                
%             else
%                 obj.ff      = flowfieldClass();     % Create flow field 
%                 fprintf('Initialized Single Flow Field with Default Settings \n')
%                 obj.plume       = filamentClass();      % Create plume (In progress)
%                 fprintf('Initialized Single Plume with Default Settings \n')
%                 obj.planner       = plannerClass();      % Create plume (In progress)
%                 fprintf('Initialized Single Planner with Default Settings \n')
            end
            obj.dt = 1/obj.Fs;
        end

        % -- Simulate functions
        function simulate(obj)
            for Plume_Sims = 1:obj.NumPlumes(1) %Plumes per simulation
                %Temporary Flowfields/Plumes to be overwritten during
                %loops. Ensures no continuation of results from prior sims.
                Temp.ff = obj.ff; 
                Temp.PLUMES = obj.plume(Plume_Sims,:);
                
                ColorSettings = {'r.','g.','b.'};

                for FF_Per_Plume = 1:obj.NumFF %Flowfiels per plume  
                    x = linspace(Temp.ff(FF_Per_Plume).xmin,Temp.ff(FF_Per_Plume).xmax,Temp.ff(FF_Per_Plume).n);
                    y = linspace(Temp.ff(FF_Per_Plume).ymin,Temp.ff(FF_Per_Plume).ymax,Temp.ff(FF_Per_Plume).m);
                    [xx,yy]=meshgrid(x,y);
                    for Frames = 1:1000 %Simulation Run
                        [Temp.ff(FF_Per_Plume), ~] = Temp.ff(FF_Per_Plume).step;
                        for Plumes_Per_FF = 1:obj.NumPlumes(2) %Plumes per flowfield
                            [Temp.PLUMES(Plumes_Per_FF), ~] = Temp.PLUMES(Plumes_Per_FF).step(Temp.ff(FF_Per_Plume));
                        end
                        
                        %TEMPORARY PLOTTING
                        if mod(Frames,10)==0
                            quiver(xx,yy, Temp.ff(FF_Per_Plume).uv(:,:,1), Temp.ff(FF_Per_Plume).uv(:,:,2))
                            hold on
                            for i = 1:obj.NumPlumes(2)
                                scatter(Temp.PLUMES(i).fil(:,1),Temp.PLUMES(i).fil(:,2),ColorSettings{i})
                            end
                            hold off
                            axis([ Temp.ff(FF_Per_Plume).xmin, Temp.ff(FF_Per_Plume).xmax, Temp.ff(FF_Per_Plume).ymin, Temp.ff(FF_Per_Plume).ymax])
                            pause(0.01)
                            
                        end
                        
                    end
                end
                Temp.ff = [];
            end
        end
        
        % -- Alternative simulation 
        function obj = simulate_v2(obj,params)
            if nargin == 1
                params.N = 1000;
                params.modnum = 10;
            end

            for k = 1: params.N
                obj = obj.step;
                if mod(k,params.modnum)==0
                    clf
                    if ~isempty(obj.terrain)
                        obj.plotterrain(1)
                    end
                    obj.plotff(1)
                    obj.plotplume(1)
                    shg
                end
            end
        end

        % -- Simulation version for GUI application
        function simulateApp(obj,app)
            for Plume_Sims = 1:obj.NumPlumes(1) %Plumes per simulation
                %Temporary Flowfields/Plumes to be overwritten during
                %loops. Ensures no continuation of results from prior sims.
                Temp.ff = obj.ff; 
                Temp.PLUMES = obj.PLUMES(Plume_Sims,:);
                
                ColorSettings = {'r.','g.','b.'};

                for FF_Per_Plume = 1:obj.NumFF %Flowfiels per plume  
                    x = linspace(Temp.ff(FF_Per_Plume).xmin,Temp.ff(FF_Per_Plume).xmax,Temp.ff(FF_Per_Plume).n);
                    y = linspace(Temp.ff(FF_Per_Plume).ymin,Temp.ff(FF_Per_Plume).ymax,Temp.ff(FF_Per_Plume).m);
                    [xx,yy]=meshgrid(x,y);
                    for Frames = 1:1000 %Simulation Run
                        [Temp.ff(FF_Per_Plume), ~] = Temp.ff(FF_Per_Plume).step;
                        for Plumes_Per_FF = 1:obj.NumPlumes(2) %Plumes per flowfield
                            [Temp.PLUMES(Plumes_Per_FF).plume, ~] = Temp.PLUMES(Plumes_Per_FF).plume.step(Temp.ff(FF_Per_Plume));
                        end
                        
                        %TEMPORARY PLOTTING
                        if mod(Frames,10)==0
                            quiver(app.UIAxes,xx,yy, Temp.ff(FF_Per_Plume).uv(:,:,1), Temp.ff(FF_Per_Plume).uv(:,:,2))
                            hold(app.UIAxes,'on')
                            for i = 1:obj.NumPlumes(2)
                                scatter(app.UIAxes,Temp.PLUMES(i).plume.fil(:,1),Temp.PLUMES(i).plume.fil(:,2),ColorSettings{i})
                            end
                            hold(app.UIAxes,'off')
                            axis([ Temp.ff(FF_Per_Plume).xmin, Temp.ff(FF_Per_Plume).xmax, Temp.ff(FF_Per_Plume).ymin, Temp.ff(FF_Per_Plume).ymax])
                            pause(0.01)
                            
                        end
                        
                    end
                end
                Temp.ff = [];
            end
        end
       
        % -- Environment Step Function
        function obj = step(obj)
            % -- step flowfield independently
            for i=1:length(obj.ff)
                [obj.ff(i),~] = obj.ff(i).step;             
            end
			% -- step filament(s) independently
            for j=1:size(obj.plume,2)
                if ~isempty(obj.terrain)
                    %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i),obj.terrain);
					[obj.plume(j),~] = obj.plume(j).step(obj.ff(i),obj.terrain);
                else
                   %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i));
					[obj.plume(j),~] = obj.plume(j).step(obj.ff(i));
                end
            end
			% for i=1:length(obj.ff)
                % [obj.ff(i),~] = obj.ff(i).step;             
                % % -- step filament
                % for j=1:size(obj.plume,2)
                    % if ~isempty(obj.terrain)
                        % %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i),obj.terrain);
						% [obj.plume(j),~] = obj.plume(j).step(obj.ff(i),obj.terrain);
                    % else
                        % %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i));
						% [obj.plume(j),~] = obj.plume(j).step(obj.ff(i));
                    % end
                % end
            % end 

            % -- step planner
            for i=1:length(obj.planner)
                % main code here
                obj.planner(i) = obj.planner(i).step(obj.ff,obj.plume);
            end

            for i = 1:obj.NumPlumes(1)
                if i==2 && ~holdtrue, hold on; end
                if obj.ff(1).dim == 3
                    obj.fil_locs = obj.plume(i).fil(:,[1:4]); 
                else 
                    obj.fil_locs = obj.plume(i).fil(:,[1,2,4]); 
                end
            end
			
        end

        % -- Add plume(s) to the environment class properties
        function obj = addPlume(obj,pobj)
            len = length(obj.plume);
            if nargin == 1
                % -- initialize default to the environment class
                obj.plume(len+1) = filamentClass();
            elseif nargin == 2
                % -- initialize an add to the environment class
				for i=1:size(pobj,2)
					obj.plume(len+i) = pobj(i);
				end
            end
            obj.NumPlumes = length(obj.plume);
        end

        % -- Remove plume(s) in the environment class
        function obj = removePlume(obj,id)
            obj.plume(id) = [];
            obj.NumPlumes = size(obj.plume);
        end

        % -- Add flowfield(s) to the environment class properties
        function obj = addFlowfield(obj,ffobj)
            len = length(obj.ff);
            if nargin == 1
                % -- initialize default to the environment class
                obj.ff(len+1) = flowfieldClass();
            elseif nargin == 2
                % -- initialize an add to the environment class
                obj.ff(len+1) = ffobj;
            end
            obj.NumFF = len+1;
        end

        % -- Remove flowfield(s) in the environment class
        function obj = removeFlowfield(obj,id)
            obj.ff(id) = [];
            obj.NumFF = length(obj.ff);
        end

        % -- Add planners(s) to the environment class properties
        function obj = addPlanner(obj,plobj)
            len = length(obj.planner);
            if nargin == 1
                % -- initialize default to the environment class
                obj.planner(len+1) = plannerClass();
            elseif nargin == 2
                % -- initialize an add to the environment class
                obj.planner(len+1) = plobj;
            end
            obj.NumPlanners = len+1;
        end

        % -- Remove planner(s) in the environment class
        function obj = removePlanner(obj,id)
            obj.planner(id) = [];
            obj.NumPlanners = length(obj.planner);
        end

        function obj = addTerrain(obj,sobj_params)
            if nargin == 2
                obj.terrain = terrainClass(sobj_params);
            else
                obj.terrain = terrainClass();
            end
        end
        % --Plotting functions
        function plotWind(obj)
            % -- get mesh for plotting
            x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
            y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            [xx,yy]=meshgrid(x,y);

            % -- plot the wind over some time
            figure(1)
            quiver(xx,yy,obj.ff.uv(:,:,1),obj.ff.uv(:,:,2))   
            for i=1:obj.ff.Fs*200
                [obj.ff,flag] = obj.ff.step();
                if mod(i,20)==0
                    quiver(xx,yy,obj.ff.uv(:,:,1),obj.ff.uv(:,:,2))
                    %axis([obj.ff.xmin,obj.ff.xmax,obj.ff.ymin,obj.ff.ymax])
                    axis([obj.xmin,obj.xmax,obj.ymin,obj.ymax])
                    %view(gca,obj.PLOTAZM+25*cos(0.005*i),obj.PLOTELV+3*cos(0.01*i))
                    view(gca,obj.PLOTAZM,obj.PLOTELV)
                    text(0,0,1,num2str(i/obj.ff.Fs))
                    shg
                    pause(0.01)
                end

            end
        end

        % --Plotting functions
        function plotPlume(obj)
            x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
            y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            [xx,yy]=meshgrid(x,y);


        end

        % --To plot all or individual flowfield at current frame
        function plotff(obj,holdtrue,ffnum,ffcolor)
            % -- get mesh for plotting
            x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
            y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            [xx,yy]=meshgrid(x,y);
            xx = xx'; yy = yy';
            c = 'k';
            if holdtrue, hold on; end
            if nargin >= 3
                i = ffnum;
                if obj.ff(i).dim == 3
                    uv_temp = squeeze(obj.ff(i).uv(1,:,:,1:2));
                    h = obj.ff.zmin;
                else
                    uv_temp = obj.ff(i).uv;
                    h = 0;
                end
				if nargin > 3
					%quiver(xx,yy,uv_temp(:,:,1),uv_temp(:,:,2),ffcolor)
                    quiver3(xx,yy,h+0*yy,uv_temp(:,:,1),uv_temp(:,:,2),0*yy,ffcolor)
				else
					quiver(xx,yy,uv_temp(:,:,1),uv_temp(:,:,2),c)
				end
            else
                for i=1:length(obj.ff)
                    if i==2 && ~holdtrue, hold on; end
                    if obj.ff(i).dim == 3
                        uv_temp = squeeze(obj.ff(i).uv(1,:,:,1:2));
                    else
                        uv_temp = obj.ff(i).uv;
                    end
                    quiver3(xx,yy,0*yy,uv_temp(:,:,1),uv_temp(:,:,2),0*yy,c)
                end
            end
            hold off
            axis([ obj.ff(1).xmin, obj.ff(1).xmax, obj.ff(1).ymin, obj.ff(1).ymax])

        end

        % -- To plot all of individual plumes at current frame
        function plotplume(obj,holdtrue,plumenum)
            % -- get mesh for plotting
            %x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
           % y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            %[xx,yy]=meshgrid(x,y);
            if holdtrue, hold on; end
            psize = 50;
            if nargin == 3
                %i = plumenum;
                %if obj.ff(1).dim == 3
                    %scatter3(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.plume(i).fil(:,3),psize,obj.ColorSettings{i})
                    %view(obj.az,obj.el)
                %else
                    %scatter(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.ColorSettings{i})
                %end
				for i = 1:obj.NumPlumes(1)
                    if i==2 && ~holdtrue, hold on; end
                    if obj.ff(1).dim == 3
                        scatter3(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.plume(i).fil(:,3),psize,obj.ColorSettings{plumenum})
                        scatter3(obj.plume(i).p0(1),obj.plume(i).p0(2),obj.plume(i).p0(3),'y*')
                        view(obj.az,obj.el)
                    else
                        scatter(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.ColorSettings{plumenum})
                    end
                end
            else
                for i = 1:obj.NumPlumes(1)
                    if i==2 && ~holdtrue, hold on; end
                    if obj.ff(1).dim == 3
                        scatter3(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.plume(i).fil(:,3),500*psize*obj.plume(i).fil(:,4),obj.ColorSettings{1})
                        scatter3(obj.plume(i).p0(1),obj.plume(i).p0(2),obj.plume(i).p0(3),'g*')
                        view(obj.az,obj.el)
                    else
                        scatter(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.ColorSettings{1})
                    end
                end
            end
			%alpha(0.5)
            hold off
            if obj.ff(1).dim == 3
                axis([ obj.ff(1).xmin, obj.ff(1).xmax, obj.ff(1).ymin, obj.ff(1).ymax, obj.ff(1).zmin, obj.ff(1).zmax*1.25])
                view(obj.az,obj.el)
            else
                axis([ obj.ff(1).xmin, obj.ff(1).xmax, obj.ff(1).ymin, obj.ff(1).ymax])
            end
        end

        function plotterrain(obj,holdtrue)
            if holdtrue, hold on; end
            X = obj.terrain.surface.XData;
            Y = obj.terrain.surface.YData;
            Z = obj.terrain.surface.ZData;
           
            colormap gray
            surf(X,Y,Z);
            shading interp;                  % Interpolate color across faces.
            camlight headlight; %left, right                   % Add a light over to the left somewhere.
            lighting gouraud;                % Use decent lighting.
            axis equal vis3d;                % Set aspect ratio.
            %axis image
            xlabel('X')
            ylabel('Y')
            zlabel('Z')
            alpha(gca,0.5)
        end

        function plotplanner(obj,holdtrue)
            if holdtrue, hold on; end
            Xg = obj.planner.Xg;
            c = obj.planner.qqm;
            n = obj.planner.np;
            for i=1: length(obj.planner.vehicle)
                p = obj.planner.vehicle(i).p;                
                s = size(p);
                if s(2) == 2
                    scatter(p(1),p(2),'gd')
                else
                    scatter3(p(1),p(2),p(3),'gd','filled')
                end
            end
            cmax = max(c);
            cmin = min(c);
            cb = 1.98;
            cs = (c-cmin)/(cmax-cmin);
            cs(cs>1) = 1;
            k = (cs>0.5);
            if cmax == 0, cmax = 1; end
            scatter3(Xg(k,1),Xg(k,2),Xg(k,3),'.','CData',[1-cs(k),1-cs(k),1-cs(k)])
%             c(c<=0)=1e-15;
%             cmin = min(c);
%             cmax = max(c);
%             if cmax > 0 && cmax ~= cmin
%                 scatter3(Xg(:,1),Xg(:,2),Xg(:,3),(c-cmin)./(cmax-cmin)+1e-15)
%             else
%                 scatter3(Xg(:,1),Xg(:,2),Xg(:,3),c)
%             end
            hold off
%             alpha(0.5)
        end

        function obj = getFrame(obj)
            frame = getframe();
            obj.frames = obj.frames + 1;
            im = frame2im(frame);
            [imind,cm] = rgb2ind(im,256);
            if obj.frames == 1
                imwrite(imind,cm,obj.fn,'gif', 'Loopcount',inf);
            else
                imwrite(imind,cm,obj.fn,'gif','WriteMode','append');
            end
        end
           
    end



    methods (Access = private)

    end

end