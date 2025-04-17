%% Planner Class
% ----------------------
% by Derek Hollenbeck
%
% This class coordinates the vehicles given a strategy and quantification method.
% The class contains vehicle class objects (including the vehicle sensor)
%
% The planner class holds the waypointList and passes the desired waypoint
% to the vehicle class for control.
%
% EXAMPLE:
%   ...
% ----------------------

classdef plannerClass

    properties
        % -- DEFAULT Planner Constants
        k               = 0;
        dt              = [];
        Fs              = 20;
        data            = struct;
        NavStrategy     = 'LawnMower';  % CVT, LawnMower, Custom
        QuantStrategy   = 'NGI'         % DE-CVT, Mod-NGI, MB, NGI, NGI+MB
        ccvt_on         = 0;
        vehicle         = vehicleClass.empty;

        delx = []; 
        dely = []; 
        delz = []; 
        dataout2 = cell(100,100,1); 
        dataout5 = cell(100,100,1); 
        xvals = []; 
        yvals = []; 
        zvals = []; 

        % -- DEFAULT Measurement variables at any given instant
        WindSpeed               = []; %Vehicle windspeed
        Concentration           = []; %Vehicle Concentration (PPM)
        Position                = []; %Vehicle Global Position
        Flux                    = []; %vehicle Flux
        Mdot                    = []; %estimated source rate SCFH;
        mdot_conv_factor        = 178; %convert to molecules/s to SCFH

        % -- DEFAULT Curtain Control Parameters
        Zlim            = [6 14];       %flux plane vertical bounds
        Width           = 10;           %flux plane horizontal width
        dZ              = 2;            %flux plane vertical path steps
        Angle           = pi/2;         %flux plane angle with respect to x-axis (in radians)
        LatLon_Position = [30 50];      %flux plane location in XY-plane
        w               = 20;           % CVT width of plane
        h               = 10;           % CVT height of plane
        r               = [40,50,5];    % CVT flux plane position vector
        rd              = [40,50,5];    % CVT desired plane position vector
        phi             = 90;           % CVT plane angle
        phid            = 90;           % CVT desired plane angle in degrees
        np              = 100;          % CVT discrete number of points for plane width
        kr              = 0.05;
        kphi            = 0.01;
        maxerr_r        = 10;

        % -- DEFAULT Quantification Parameters (might turn into class
        % later)
		STOREMEASUREMENTS  	  	= 0;
        p                       = []; % stored position
        c                       = []; % stored concentration
        ws                      = []; % stored ws speed
        qm                      = []; % measured flux
        Q                       = []; % estimated source rate
        sigy                    = []; % estimated dispersion in y
        sigz                    = []; % estimated dispersion in z
        Ty                      = []; % estimated scale factor in y
        Tz                      = []; % estimated scale factor in z
        x_hat                   = 10; % estimated distance to the source
        P_hat                    = [1,4,4,0,5]; % stored estimated params

        % -- DEFAULT NGI Optimization Parameters
        options                 = optimset();
        Qbnd                    = [1e-6 50];
        Tbnd                    = [0.03 0.5];
        Sbnd                    = [1,30];
        Q_hat                   = 4*1e-4; % default source estimate guess (for optimization)
        distType                = 8;

        % -- Kriging Parameters
        distTypes               = {'blinear','circular','spherical','pentaspherical',...
            'exponential','gaussian','whittle','stable','matern'};
        lag                     = 15;
        alph                    = 0.7;

        % -- CVT Parameters
        kp                      = 5;%1;
        kpe                     = 0;
        kv                      = 0;
        tau                     = [];   % Delaunay Triangulation Points
        winlen                  = 100;  % length of samples used for quantification
        Xl                      = [];   % local plane coordinates
        Xg                      = [];   % global plane coordinates
        qqm                     = [];

        % -- Conversion Parameters
        Na = 6.0221409*10^23;     % Avagadros number
        M = 16.04;                % molecular mass
        bkg_ch4 = 1.98;           % background methane
        T = 273;                  % K, temperature
        R = 8.31446;              % J / mol K
        P = 101325;               % Pa
        No = [];                  % Air number density

        % -- function handles
        J = @(q,q_hat) sum((q-q_hat).^2);
        f = @(x,theta) theta(1)/(2*pi*theta(2)*theta(3)) .*...
            exp(-0.5*( (x(:,1)-theta(4)) / theta(2) ).^2) .*...
            (exp(-0.5*( (x(:,2)+theta(5)) / theta(3) ).^2) + exp(-0.5*( (x(:,2)-theta(5)) / theta(3) ).^2));

    end

    properties (Access = private)
        SingleUAV_Waypoints_ID = 1;
        SingleUAV_Waypoints_flag = [];
        numVehicles = 0;
        Quantified  = 0;
    end

    methods
        % -- CONSTRUCTOR
        function obj = plannerClass(params)
            % -- Update planner default strategy, quantification,
            if nargin == 1
                % -- Plane params
                if isfield(params,'Zlim'),  obj.Zlim    = params.Zlim;  end
                if isfield(params,'Width'), obj.Width   = params.Width; end
                if isfield(params,'dZ'),    obj.dZ      = params.dZ;    end
                if isfield(params,'Angle'), obj.Angle   = params.Angle; end
                if isfield(params,'LatLon_Position'), obj.LatLon_Position = params.LatLon_Position;  end
                if isfield(params,'r'), obj.r   = params.r; end
                if isfield(params,'rd'), obj.rd = params.rd;  end
                if isfield(params,'w'), obj.w   = params.w; end
                if isfield(params,'xvals'), obj.xvals   = params.xvals; end
                if isfield(params,'yvals'), obj.yvals   = params.yvals; end
                if isfield(params,'zvals'), obj.zvals   = params.zvals; end
                if isfield(params,'dataout2'), obj.dataout2   = params.dataout2; end
                if isfield(params,'dataout5'), obj.dataout5   = params.dataout5; end
                if isfield(params,'h'), obj.h = params.h;  end
                if isfield(params,'phi'), obj.phi   = params.phi; end
                if isfield(params,'phid'), obj.phid = params.phid;  end
                % -- Strategies
                if isfield(params,'NavStrategy'), obj.NavStrategy = params.NavStrategy; end
                if isfield(params,'QuantStrategy'), obj.QuantStrategy = params.QuantStrategy; end
                % -- Hyperparams for strategies
                %if isfield(params,'Zlim'),  obj.Zlim    = params.Zlim;  end
                % -- Vehicle params
                if isfield(params,'vehicleParams')
                    obj.vehicle    = vehicleClass(params.vehicleParams);
                    obj.numVehicles = 1;
                    fprintf('Initialized 1 vehicle \n')
                else
                    obj.vehicle = vehicleClass();
                    obj.numVehicles = 1;
                    obj.vehicle = obj.vehicle.setWaypoints(obj.getPlaneWaypoints());
                    fprintf('Initialized 1 vehicle with Default Settings \n')
                end
                % -- Other
            else
                %                 obj.vehicle = vehicleClass();
                %                 obj.vehicle = obj.vehicle.setWaypoints(obj.getPlaneWaypoints());
                %                  fprintf('Initialized 1 vehicle with Default Settings \n')
                obj.vehicle = vehicleClass();
                obj.numVehicles = 1;
                obj.vehicle = obj.vehicle.setWaypoints(obj.getPlaneWaypoints());
                fprintf('Initialized 1 vehicle with Default Settings \n')
            end
            % --  Calculated Default params
            obj.dt = 1/obj.Fs;                  % time step
            obj.No = obj.P*obj.Na/obj.T/obj.R;  % air number density
            [xx,zz] = meshgrid(linspace(-obj.w/2,obj.w/2,obj.np), ... % local plane coordinates
                linspace(0,obj.h,obj.np));
            yy = 0*xx;
            obj.Xl = [xx(:),yy(:),zz(:)];       % store local coord. in obj
            obj.Xg = obj.getPlaneGlobal(obj.Xl,obj.r,obj.phi); % store glocal coord. in obj
            obj.options.MaxFunEvals = 1000;
            obj.options.MaxIter = 1000;
            obj.options.TolFun = 1e-9;
            obj.options.TolX = 1e-9;
            obj.options.Display = 'off';
        end

        % -- STEP FUNCTION
        function obj = step(obj,ff,plume)
            obj.k = obj.k + 1;
            % -- update vehilce location if needed.
            n = obj.numVehicles;
			nplumes = length(plume);

           
            Data = obj.vehicle(1); 
            

            % -- take recent measurements
            % for i = 1:n
			% 	for j = 1:nplumes
				% 	% -- take each vehicle's measurements
				% 	obj.vehicle(i) = obj.measure(obj.vehicle(i),ff,plume);
				% 	obj.Concentration(i,:) = obj.vehicle(i).C;
				% 	obj.Position(i,:) = obj.vehicle(i).p;
				% 	obj.WindSpeed(i,:) = obj.vehicle(i).ws;
			% 	end
            % end

            for i = 1:100
                %for j = 1:400 
                    for kk = [1,2]
                    %for kk = 1
                        coor = [obj.xvals(i).*ones(100,1),obj.yvals',obj.zvals(kk).*ones(100,1)]; 
                        Data.p = coor; 
                        %Data = Data.measure2(Data,ff,plume); 
                        Data = obj.measure3(Data,ff,plume); 
                        vals = [Data.ws Data.C]; 
                        if kk == 1
                            obj.dataout2(i,1:100) = num2cell(vals,2)';
                        else 
                            obj.dataout5(i,1:100) = num2cell(vals,2)';
                        end
                    end

                %end
            end

            %for i = 1:400
            %    for j = 1:400 
            %        for kk = 2
            %            coor = [obj.xvals(i),obj.yvals(j),obj.zvals(kk)]; 
            %            Data.p = coor; 
            %            %Data = Data.measure2(Data,ff,plume); 
            %            Data = obj.measure2(Data,ff,plume); 
            %            vals = [Data.ws Data.C]; 
            %            obj.dataout5{i,j,1} = vals; 
            %        end
            %    end
            %end


            switch obj.NavStrategy
                
                case 'CVT' % -- MultiUAS
                    % -- compute flux for all vehicles based on normal
                    
%                     dim = size(obj.Position,2);
%                     if dim == 3
%                         coef = pca(obj.Position);
%                         nhat = coef(:,3);
%                     else
%                         nhat = [0,1,0];
%                     end
%                     %nhat = [0,0,1]'; 
%                     nhat = (nhat*ones(1,n))';
%                     Vdot = abs(dot(nhat,obj.WindSpeed,2));
%                     obj.Flux = (obj.Concentration-1.98)*1e-6*obj.P*obj.M/(obj.R*obj.T).*Vdot;  % need to subtract background if avail.
% 
%                     % -- store measurements for computing
%                     %obj.p = [obj.p; obj.Position];
%                     %obj.c = [obj.c; obj.Concentration];
%                     %obj.ws = [obj.ws; obj.WindSpeed];
%                     %obj.qm = [obj.qm; obj.Flux];
% 
%                     % -- truncate memory if over window length
%                     %if length(obj.c) > obj.winlen
%                     %obj.p = obj.p(end-obj.winlen:end,:);
%                     %obj.c = obj.c(end-obj.winlen:end);
%                     %obj.ws = obj.ws(end-obj.winlen:end,:);
%                     %obj.qm = obj.qm(end-obj.winlen:end,:);
%                     %end
% 
%                     % -- store measurements for computing
%                     obj.p = [obj.Position; obj.p];
%                     obj.c = [obj.Concentration; obj.c];
%                     obj.ws = [obj.WindSpeed; obj.ws];
%                     obj.qm = [obj.Flux; obj.qm];
% 
%                     % -- truncate memory if over window length
%                     if length(obj.c) > obj.winlen
%                         obj.p = obj.p(1:obj.winlen,:);
%                         obj.c = obj.c(1:obj.winlen);
%                         obj.ws = obj.ws(1:obj.winlen,:);
%                         obj.qm = obj.qm(1:obj.winlen,:);
%                     end
% 
%                     % -- perform quantification algorithm
%                     switch obj.QuantStrategy
%                         case 'Mod-NGI'
%                             % -- Convert global positions to local
%                             p = obj.getPlaneLocal(obj.p,obj.r,obj.phi); % get xz points
% 
%                             if 0 % regular CVT or CCVT
%                                 % -- Compute modNGI MLE param estimates
%                                 Ns = length(obj.qm);
%                                 mu_hat(1) = sum(obj.qm.*p(:,1))/sum(obj.qm);
%                                 mu_hat(2) = sum(obj.qm.*p(:,3))/sum(obj.qm);
%                                 sig_hat(1) = norm(sqrt( sum(obj.qm.*(p(:,1)-mu_hat(1)).^2) / ((Ns-1)*sum(obj.qm)/Ns) ));
%                                 sig_hat(2) = norm(sqrt( sum(obj.qm.*(p(:,3)-mu_hat(2)).^2) / ((Ns-1)*sum(obj.qm)/Ns) ));
%                                 % -- Compute Optimized parameters
%                                 % run optimization
%                                 %                             P_hat = fminsearchbnd(@(P) Ns^-1 * sum( (obj.qm - obj.f([p(:,1),p(:,3)],P)).^2 ) + (obj.lambda/(2*Ns)).*sum(abs(P)),...
%                                 if mod(obj.k,round(obj.winlen/obj.numVehicles))
%                                     P_hat = fminsearchbnd(...
%                                         @(P) Ns^-1 * sum( (obj.qm - obj.f([p(:,1),p(:,3)],P)).^2 ),...  % function
%                                         [obj.Q_hat,sig_hat,mu_hat],...                                  % initial guess
%                                         [obj.Qbnd(1),obj.Sbnd(1),obj.Sbnd(1),-obj.w/2,0],...            % LB
%                                         [obj.Qbnd(2),obj.Sbnd(2),obj.Sbnd(2),obj.w/2,obj.h],...         % UB
%                                         obj.options);                                                   % options
%                                     P_hat = real(P_hat);
%                                     obj.P_hat = P_hat;
%                                 else
%                                     P_hat = obj.P_hat;
%                                 end
% 
%                                 % -- Compute NGI control points
%                                 P_hatc = [
%                                     P_hat(1),...
%                                     P_hat(2),...
%                                     P_hat(3),...
%                                     P_hat(4) + 1*cos(0.2*obj.k*obj.dt),...
%                                     P_hat(5) + 1.2*sin(0.1*obj.k*obj.dt)...
%                                     ];
% 
%                             else
%                                 % -- Compute modNGI MLE param estimates
%                                 Ns = length(obj.qm);
%                                 mu_hat(1) = sum(obj.qm.*p(:,1))/sum(obj.qm);
%                                 mu_hat(2) = sum(obj.qm.*p(:,3))/sum(obj.qm);
%                                 sig_hat(1) = norm(sqrt( sum(obj.qm.*(p(:,1)-mu_hat(1)).^2) / ((Ns-1)*sum(obj.qm)/Ns) ));
%                                 sig_hat(2) = norm(sqrt( sum(obj.qm.*(p(:,3)-mu_hat(2)).^2) / ((Ns-1)*sum(obj.qm)/Ns) ));
%                                 %P_hat = [obj.Q_hat,sig_hat,mu_hat];
% 
%                                 % -- Compute Optimized parameters
%                                 % run optimization
%                                 %                             P_hat = fminsearchbnd(@(P) Ns^-1 * sum( (obj.qm - obj.f([p(:,1),p(:,3)],P)).^2 ) + (obj.lambda/(2*Ns)).*sum(abs(P)),...
%                                 %if mod(obj.k,round(obj.winlen/obj.numVehicles))
%                                 if mod(obj.k,round(5*obj.Fs))
%                                     P_hat = fminsearchbnd(...
%                                         @(P) Ns^-1 * sum( (obj.qm - obj.f([p(:,1),p(:,3)],P)).^2 ),...  % function
%                                         [obj.Q_hat,sig_hat,mu_hat],...                                  % initial guess
%                                         [obj.Qbnd(1),obj.Sbnd(1),obj.Sbnd(1),-obj.w/2,0],...            % LB
%                                         [obj.Qbnd(2),obj.Sbnd(2),obj.Sbnd(2),obj.w/2,obj.h],...         % UB
%                                         obj.options);                                                   % options
%                                     P_hat = real(P_hat);
%                                     obj.P_hat = P_hat;
%                                 else
%                                     P_hat = obj.P_hat;
%                                 end
% 
%                                 % -- Compute NGI control points
%                                 switch 1
%                                     case 1
%                                         P_hatc = [
%                                             obj.Q_hat,...
%                                             obj.w/4,...
%                                             obj.h/4,...
%                                             0 + 0*cos(0.2*obj.k*obj.dt),...
%                                             obj.h/2 + 0*sin(0.1*obj.k*obj.dt)...
%                                             ];
%                                     case 2
%                                         P_hatc = [
%                                             P_hat(1),...
%                                             P_hat(2),...
%                                             P_hat(3),...
%                                             P_hat(4) + 0*cos(0.2*obj.k*obj.dt),...
%                                             P_hat(5) + 0*sin(0.1*obj.k*obj.dt)...
%                                             ];
%                                 end
% 
%                             end
% 
%                             obj.Mdot = obj.mdot_conv_factor*obj.P_hat(1);
%                             %obj.Phat_t = [P_hat;obj.Phat_t];
% 
% 
%                             % -- Compute Delauny triangulation
%                             %tau = delaunay ( p(end-obj.numVehicles:end,1) + 0.01*rand(swarm_num,1), p(end-obj.numVehicles:end,3) + 0.01*rand(swarm_num,1) );
%                             obj.tau = delaunay(p(1:n,1) + 0.1*randn(n,1), p(1:n,3) + 0.1*randn(n,1));
%                             k = dsearchn ( [p(1:n,1),p(1:n,3)], obj.tau, [obj.Xl(:,1),obj.Xl(:,3)] );
% 
%                             % -- Compute centroids
%                             % find centroids
%                             qqm = obj.f([obj.Xl(:,1),obj.Xl(:,3)],P_hatc);
%                             c = zeros(n,dim)+1e-15;
%                             qqm(qqm<1e-15) = 1e-15;
%                             obj.qqm = qqm;
%                             try
%                                 M(1:n,1) = accumarray ( k, qqm );
%                             catch
%                                 M(1:n,1) = 1e-15;
%                                 fprintf('error in accumarray')
%                             end
%                             M(M<1e-15) = 1e-15;
%                             Mxy(1:n,1) = accumarray ( k, qqm.*obj.Xl(:,1) ); % mass * location: x * f(x,y)
%                             Mxy(1:n,3) = accumarray ( k, qqm.*obj.Xl(:,3) ); % mass * location: y * f(x,y)
%                             c(1:n,1) = Mxy(1:n,1) ./ M(1:n); % center of mass, ci
%                             c(1:n,3) = Mxy(1:n,3) ./ M(1:n); % center of mass, cj
% 
%                             % -- Continuous CVT mode
%                             if obj.ccvt_on
%                                 cv = [];
%                                 X = obj.getPlaneLocal(obj.Position,obj.r,obj.phi); 
%                                 X(:,2)=[]; 
%                                 xc = [0,obj.h/2];
%                                 vmax = 1;
%                                 lam = 1;
%                                 gam = -0.5;
%                                 bet = -0.1;
%                                 % A = eye(2);
%                                 a11 = bet;
%                                 a12 = 1;
%                                 a21 = gam;
%                                 a22 = bet;
%                                 A = [a11,a12;a21,a22];
%                                 cv = obj.dt*((lam*A/norm(A))*(X-xc)')'+X;
%                                 cv = [cv(:,1),zeros(n,1),cv(:,2)];
%                                 if 0
%                                     scatter(c(:,1),c(:,3),'r')
%                                     hold on
%                                     scatter(cv(:,1),cv(:,3),'b')
%                                     scatter((cv(:,1)+c(:,1))/2,(cv(:,3)+c(:,3))/2,'g')
%                                     scatter(X(:,1),X(:,2),'k*')
%                                     hold off
%                                     axis([-obj.w/2,obj.w/2,0,obj.h])
%                                 end
% 
%                                 % check the boundaries
%                                 cv(cv(:,3)<0,3) = 0;
%                                 cv(cv(:,3)>obj.h,3) = obj.h;
%                                 cv(cv(:,1)>obj.w/2,1) = obj.w/2;
%                                 cv(cv(:,1)<-obj.w/2,1) = -obj.w/2;
% 
%                                 % check the boundaries
%                                 c(c(:,3)<0,3) = 0;
%                                 c(c(:,3)>obj.h,3) = obj.h;
%                                 c(c(:,1)>obj.w/2,1) = obj.w/2;
%                                 c(c(:,1)<-obj.w/2,1) = -obj.w/2;
% 
%                                 temp = obj.getPlaneGlobal((c+cv)./2,obj.r,obj.phi);
% 
%                             else
%                                 % check the boundaries
%                                 c(c(:,3)<0,3) = 0;
%                                 c(c(:,3)>obj.h,3) = obj.h;
%                                 c(c(:,1)>obj.w/2,1) = obj.w/2;
%                                 c(c(:,1)<-obj.w/2,1) = -obj.w/2;
%                                 temp = obj.getPlaneGlobal(c,obj.r,obj.phi);
%                             end
% 
%                             % -- Update vehicle waypoints
% %                             temp = obj.getPlaneGlobal(c,obj.r,obj.phi);
% %                             temp = obj.getPlaneGlobal((c+cv)./2,obj.r,obj.phi);
% %                             temp = obj.getPlaneGlobal(cv,obj.r,obj.phi);
%                             for i = 1:obj.numVehicles
%                                 % -- take each vehicle's measurements
%                                 obj.vehicle(i) = obj.vehicle(i).setWaypoints(temp(i,:));
%                             end
% 
%                         otherwise
%                             fprintf('error in planner step function')
%                     end
% 
%                     % -- step each vehicle
%                     for i = 1:obj.numVehicles
%                         obj.vehicle(i) = obj.vehicle(i).step();
%                     end
% 
%                     % -- perform some plane control here
%                     % < insert code >
% 
%                     % -- step plane
%                     obj = obj.stepPlane;
% 
%                 case {'LawnMower', 'Custom'} % -- SingleUAS
%                     % -- store measurements for computing
% 					if (obj.STOREMEASUREMENTS)
% 						obj.p = [obj.Position; obj.p];
% 						obj.c = [obj.Concentration; obj.c];
% 						obj.ws = [obj.WindSpeed; obj.ws];
% 					else
% 						obj.p = obj.Position;
% 						obj.c = obj.Concentration;
% 						obj.ws =obj.WindSpeed;
% 					end
%                     % -- step each vehicle
%                     for i = 1:obj.numVehicles
%                         obj.vehicle(i) = obj.vehicle(i).step();
% 
%                         if (obj.vehicle(i).QuantifyFlag == 1) && (obj.Quantified == 0)
%                             p = obj.getPlaneLocal(obj.p,obj.r,obj.phi); % get xz points
%                             % -- compute flux for all vehicles based on normal
%                             dim = size(obj.Position,2);
%                             if dim == 3
%                                 coef = pca(p);
%                                 nhat = coef(:,3);
%                             else
%                                 nhat = [0,1,0];
%                             end
%                             nhat = (nhat*ones(1,n))';
%                             Vdot = abs(dot(nhat.*ones(size(obj.ws,1),1),obj.ws,2));
%                             obj.qm = (obj.c-1.98)*1e-6*obj.P*obj.M/(obj.R*obj.T).*Vdot;  % need to subtract background if avail.
% 
%                             %                             %Convert from ppm
%                             %                             C = obj.Concentration*10^-6 *...
%                             %                                 (filament.P*filament.M)/(filament.R*filament.T);
% 
%                             %coordinates
%                             obj = obj.NGI(p(:,1), p(:,3),1000);
%                             obj.Quantified = 1;
%                         end
% 
%                     end
% 
% 
% 
%                 case 'Fluxotaxis'
%                     switch obj.QuantStrategy
%                         case 'GDMF'
% 
%                         otherwise
%                     end
            end

        end

        % -- STEP PLANE FUNCTION
        function obj = stepPlane(obj)
            % update plane position
            kr = obj.kr; %0.01;
            kphi = obj.kphi; %0.1;
            er = obj.rd - obj.r;
            if norm(er) > obj.maxerr_r
                er = obj.maxerr_r*er/norm(er);
            end
            ephi = obj.phid - obj.phi;
            if ephi > 180
                ephi = ephi - 180;
            elseif ephi < -180
                ephi = ephi + 180;
            end
            obj.r = obj.r + obj.dt * (kr*er);
            obj.phi = obj.phi + obj.dt * (kphi*ephi);
            % compute new plane points
            obj.Xg = obj.getPlaneGlobal(obj.Xl,obj.r,obj.phi);
        end

        % -- MEASUREMENT FUNCTION
        function vobj = measure2(obj,vobj,ff,plume)
            p = vobj.p;
            for i=1:length(vobj.sensor)
                alph = vobj.sensor(i).alph;
                dt = 1/obj.Fs;
                switch vobj.sensor(i).ID
                    case 1
                        %if ~isempty(vobj.ws)
                        %    ws_old = vobj.ws(end,:);
                        %    ws = ws_old - alph * dt * (ws_old - ff.wind_measurement(p)) + vobj.sensor(i).eta*randn();
                        %else
                        ws = ff.wind_measurement(p) + vobj.sensor(i).eta*randn();
                        %end
                    case 2
						%for j=1:length(plume)
						%	if ~isempty(vobj.C)
						%		Cold = vobj.C(j);
								% C(j) = Cold - alph * dt * (Cold - plume(j).concentration_measurement(p) - 1.98) + vobj.sensor(i).eta*randn();
                        %        C(j) = Cold - alph * dt * (Cold - plume(j).concentration_measurement(p) - obj.bkg_ch4) + vobj.sensor(i).eta*randn();
						%	else
								% C(j) = plume(j).concentration_measurement(p) + vobj.sensor(i).eta*randn() + 1.98;
                        C(1) = plume(1).concentration_measurement2(p) + vobj.sensor(i).eta*randn() + obj.bkg_ch4;
						%	end
						%end
                end
            end
            vobj.C = C;
            vobj.ws = ws;
        end

        function vobj = measure3(obj,vobj,ff,plume)
            p = vobj.p;
            for i=1:length(vobj.sensor)
                alph = vobj.sensor(i).alph;
                dt = 1/obj.Fs;
                switch vobj.sensor(i).ID
                    case 1
                        %if ~isempty(vobj.ws)
                        %    ws_old = vobj.ws(end,:);
                        %    ws = ws_old - alph * dt * (ws_old - ff.wind_measurement(p)) + vobj.sensor(i).eta*randn();
                        %else
                        ws = ff.wind_measurement(p) + vobj.sensor(i).eta*randn(size(p,1),3);
                        %end
                    case 2
						%for j=1:length(plume)
						%	if ~isempty(vobj.C)
						%		Cold = vobj.C(j);
								% C(j) = Cold - alph * dt * (Cold - plume(j).concentration_measurement(p) - 1.98) + vobj.sensor(i).eta*randn();
                        %        C(j) = Cold - alph * dt * (Cold - plume(j).concentration_measurement(p) - obj.bkg_ch4) + vobj.sensor(i).eta*randn();
						%	else
								% C(j) = plume(j).concentration_measurement(p) + vobj.sensor(i).eta*randn() + 1.98;
                        C = plume(1).concentration_measurement2(p).' + vobj.sensor(i).eta*randn(size(p,1),1) + obj.bkg_ch4;
						%	end
						%end
                end
            end
            vobj.C = C;
            vobj.ws = ws;
        end



        % -- MEASUREMENT FUNCTION
        function vobj = measure(obj,vobj,ff,plume)
            p = vobj.p;
            for i=1:length(vobj.sensor)
                alph = vobj.sensor(i).alph;
                dt = 1/obj.Fs;
                switch vobj.sensor(i).ID
                    case 1
                        if ~isempty(vobj.ws)
                            ws_old = vobj.ws(end,:);
                            ws = ws_old - alph * dt * (ws_old - ff.wind_measurement(p)) + vobj.sensor(i).eta*randn();
                        else
                            ws = ff.wind_measurement(p) + vobj.sensor(i).eta*randn();
                        end
                    case 2
						for j=1:length(plume)
							if ~isempty(vobj.C)
								Cold = vobj.C(j);
								% C(j) = Cold - alph * dt * (Cold - plume(j).concentration_measurement(p) - 1.98) + vobj.sensor(i).eta*randn();
                                C(j) = Cold - alph * dt * (Cold - plume(j).concentration_measurement(p) - obj.bkg_ch4) + vobj.sensor(i).eta*randn();
							else
								% C(j) = plume(j).concentration_measurement(p) + vobj.sensor(i).eta*randn() + 1.98;
                                C(j) = plume(j).concentration_measurement(p) + vobj.sensor(i).eta*randn() + obj.bkg_ch4;
							end
						end
                end
            end
            vobj.C = C;
            vobj.ws = ws;
        end

        % -- GET PLANE WAYPOINTS
        function waypoints = getPlaneWaypoints(obj,params)
            if nargin == 1
                switch obj.NavStrategy
                    case {'LawnMower', 'Custom'} % -- SingleUAS
                        waypoints = obj.LawnMowerWaypoints();
                    case {'CVT'}
                        %waypoints = obj.getCVTWaypoints
                        waypoints = obj.LawnMowerWaypoints();
                end
            else
                %                 obj.Waypoints = Waypoints;
                %                 obj.Waypoints_ID = 1;
            end
        end

        % -- ADD VEHICLE
        function obj = addVehicle(obj,vobj)
            len = length(obj.vehicle);
            obj.vehicle(len+1) = vobj;
            obj.numVehicles = len + 1;
        end

        % -- GET VEHICLE
        function vobj = getVehicle(obj,vobj_num)
            vobj = obj.vehicle(vobj_num);
        end

        % -- REMOVE VEHICLE
        function obj = removeVehicle(obj,vnum)
            obj.vehicle(vnum) = [];
        end

        % -- GET CVT WAYPOINTS
        function waypoints = getCVTWaypoints(obj)
            % return CVT control waypoints

        end

        % -- SUPPORT FUNCTIONS
        % Get global coordinates from local
        function Xg = getPlaneGlobal(obj,Xl,r,phi)
            c = cosd(phi);
            s = sind(phi);
            DCM = [c,s,0;-s,c,0;0,0,1];
            Xg = Xl*DCM + r;
        end

        % Get local coordinates from global
        function Xl = getPlaneLocal(obj,Xg,r,phi)
            c = cosd(-phi);
            s = sind(-phi);
            DCM = [c,s,0;-s,c,0;0,0,1];
            Xl = [Xg-r]*DCM;
        end





    end
    %Insert Strategies here


    methods (Access = private)
        function obj = Quantify(obj, ff, filament)
            %             switch obj.NAVStrategy
            %                 case {'CVT'}
            switch obj.QuantStrategy
                case 'DE-CVT'

                case 'mod-NGI'

                    %                     end
                    %                 case {'LawnMower'}
                    %                     switch obj.QuantStrategy
                case {'MB'}

                case {'NGI'}
                    %eventually dot windspeed with surface
                    %normal

                    %Get orthogonal wind speed to plane
                    ws_norm = dot(obj.WindSpeed ,...
                        [ff.uv0 0].*ones(size(ws,1),1),2);
                    %Convert from ppm
                    C = obj.Concentration*10^-6 *...
                        (filament.P*filament.M)/(filament.R*filament.T);
                    %Calculate Flux
                    obj.qm = obj.Concentration.*ws_norm;
                    %Need to transform global coordinates to local
                    ym = obj.Position(:,2); zm = obj.Position(:,3);
                    %coordinates
                    [obj.Q, obj.Tz] = NGI(obj.qm,obj.x_hat,ym,zm,obj.Qbnd,obj.Tbnd,obj.Q_hat);
            end
            %             end
        end
        % -- NAVIGATION STRATEGIES
        function UAV_Waypoints = LawnMowerWaypoints(obj)
            % -- Function for rotations about Z-axis
            R = @(Gamma) [cos(Gamma) -sin(Gamma) 0; sin(Gamma) cos(Gamma) 0; 0 0 1];

            % -- Get Z positions of flight path
            Z_int = repelem(obj.Zlim(1):obj.dZ:obj.Zlim(2),2);

            % -- Get X positions of flight path
            X_int = (-1).^(1:((obj.Zlim(2)-obj.Zlim(1))/obj.dZ)+1);
            X_int = [-obj.Width./2; obj.Width./2]*X_int;
            X_int = X_int(:);

            % -- Compile flight path points in flight plane's point of reference
            Waypoints = [X_int,zeros(size(X_int)),Z_int'];

            % -- Transform flight path points into Terrain point of reference
            UAV_Waypoints = (Waypoints*R(obj.Angle)) + ...
                [obj.LatLon_Position 0];
        end

        %         function obj = WaypointControl_Step(obj)
        %             try
        %                 obj.vehicle.Waypoint = obj.Waypoints(obj.Waypoints_ID,:);
        %                 [obj.vehicle, obj.Waypoints_flag]     = obj.vehicle.step();
        %
        %                 %After reaching a current waypoint, start going to
        %                 %next waypoint
        %                 if (obj.Waypoints_flag == 1) && (obj.Waypoints_ID < size(obj.Waypoints,1))
        %                     obj.Waypoints_ID = obj.Waypoints_ID + 1;
        %                 end
        %
        %                 %After passing the first waypoint, do all of this
        %                 if obj.Waypoints_ID > 1
        %
        %                     %Flag being ready to quantify once the final
        %                     %waypoint is reached
        %                     if (obj.Waypoints_ID ==...
        %                             size(obj.Waypoints,1)) && ...
        %                             (obj.Waypoints_flag == 1)
        %
        %                         obj.QuantifyFlag = 1;
        %                     else
        %                         % -- Store these values only when in the plane
        %                         obj.WindSpeed       = [obj.WindSpeed; obj.vehicle.ws];
        %                         obj.Position        = [obj.Position; obj.vehicle.p];
        %                         obj.Concentration   = [obj.Concentration; obj.vehicle.C];
        %                     end
        %                 end
        %                 %If at the final waypoint and Fe isn't calculates
        %                 if obj.QuantifyFlag == 1 && obj.Fe == []
        %                     obj = Quantify(obj, ff, filament);
        %                 end
        %             catch
        %                 disp('Error: Too many UAVs')
        %             end
        %         end
        %% QUANTIFICATION STRATEGIES
        function Q_est = MB(obj,filament,qm,x,z)
            xmax = max(x);
            zmax = max(z);
            [X,Z]   = meshgrid(linspace(-xmax,xmax,obj.np),linspace(0,zmax,obj.np));
            qmax = max(qm);
            qnorm = qm/qmax; % normalize
            %Chat2 = (Chat - minChat)./(maxChat - minChat);   % normalize
            %     Chat2 = Chat2 - 0.5;    % mean shift
            %     Chat2 = Chat;

            %     Chat2(isnan(Chat2))=0;
            v = variogram([x, z],qnorm,'plotit',false,'maxdist',cvt.lag);
            temp = find(isnan(v.val));
            if ~isempty(temp)
                v.val(temp) = [];
                v.distance(temp) = [];
                v.num(temp) = [];
            end

            switch obj.distType
                case 8
                    [~,~,~,vstruct] = variogramfit(v.distance,v.val,[],[],[],...
                        'model','stable','stablealpha',obj.alph,'plotit',false);
                otherwise
                    [~,~,~,vstruct] = variogramfit(v.distance,v.val,[],[],[],...
                        'model',obj.distTypes{obj.distType},'plotit',false);
            end
            qnorm(isnan(qnorm))=1e-10*randn();
            [qkrigged,qkriggedVar] = kriging(vstruct,x,z,qnorm,X,Z);
            qkrigged = qkrigged*qmax;
            qkriggedVar = qkriggedVar*qmax;

            mdot = (35.3147*60*60/0.688)*sum(qkrigged*1e-6,'all')*2*xmax*...
                zmax*filament.airNumberDensity * filament.M / filament.Na / 1e3 / 1e2;
            Q_est = mdot;
        end

        function q = ngi_forward(obj,Q,x,y,z,yc,zc,Ty,Tz) %Forward GP-Model
            %Equation 7
            q = (Q./(2*pi*Ty*Tz*x.^2)).*exp(-(y-yc).^2 ./ (2*(Ty*x).^2)).*...
                (exp(-(z-zc).^2 ./ (2*(Tz*x).^2))+exp(-(z+zc).^2 ./ (2*(Tz*x).^2)));
        end

        function q =  ngi_backward(obj,qm,Q,x,yz,Tz,zc) %Backward GP-Model
            % MEASUREMENT LOCATIONS
            y = yz(:,1);
            z = yz(:,2);
            % COMPUTE CENTER OF MASS IN y-axis
            qm_y = qm .* x .* sqrt(2*pi) ./ ...
                ((exp(-(z-zc).^2 ./ (2*(Tz*x).^2))...
                +exp(-(z+zc).^2 ./ (2*(Tz*x).^2))));

            yc = sum(qm_y.*y)/sum(qm_y);

            % COMPUTE tau_y
            Ty = norm(sqrt((sum(qm_y.*((y-yc)./x).^2,'all'))/sum(qm_y,'all')));

            % COMPUTE OUTPUT q
            q = obj.ngi_forward(Q,x,y,z,yc,zc,Ty,Tz);
        end

        function obj = NGI(obj,ym,zm,N)
            %%NGI - Shah
            % qm - Measured methane flux
            % x  - Distance to source
            % (ym,zm)    - Sample location in flux plane
            % N  - Number of steps in Tz domain
            % Qbnd  - Bounds on Q optimization (1x2 vector)
            % Tbnd  - Bounds on Tz optimization (1x2 vector)
            % Q_hat  - Initial Fe guess
            xm = [ym,zm];
            zc = sum(obj.qm.*zm)/sum(obj.qm);
            Tub = linspace(obj.Tbnd(1),obj.Tbnd(2),N);
            Tz = [];
            Q = [];

            for i = 1:length(Tub)
                %Optimize to find Fe and Tz
                options = optimoptions("lsqcurvefit",...
                    "Algorithm","levenberg-marquardt",...
                    "FunctionTolerance",10^-10,...
                    "StepTolerance",10^-16);
                options.OptimalityTolerance;

                lb = [obj.Qbnd(1) 0.0001];
                ub = [obj.Qbnd(2)  Tub(i)];
                P0 = [obj.Q_hat Tub(i)];

                Param = lsqcurvefit( ...
                    @(P,yz) obj.ngi_backward(obj.qm,P(1),obj.x_hat,yz,P(2),zc),...
                    P0,xm,obj.qm,lb,ub,...
                    options);

                Tz(i) = Param(2);
                Q(i) = Param(1);
            end
            for m = 2:length(Tz)
                if (Tz(m)-Tz(m-1))^2<=1e-10 && (Q(m)-Q(m-1))^2<=1e-10
                    obj.Tz = Tz(m);
                    obj.Q = Q(m);
                    break
                end
                if m == length(Tz) && isempty(Tz_est)
                    obj.Tz = NaN;
                    obj.Q = NaN;
                end
            end

        end

    end

end