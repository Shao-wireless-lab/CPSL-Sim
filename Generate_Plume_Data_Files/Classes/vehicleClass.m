%% UAV Class
% ----------------------
% by Derek Hollenbeck
%
% This class creates an environment object that acts as a container for
% running the MOABS/DT simulation
%
% EXAMPLE:
%   ...
% ----------------------
classdef vehicleClass

    properties
        % default params and variables

        % -- DEFAULT UAV Constants
        p               = [25 25 10];
        v               = [0 0 0];
        C               = [];
        ws              = [];
        dt              = [];
        Fs              = 20;
        alim            = 1;
        vlim            = 2;
        Kp              = 0.5+2;
        Kd              = 1.65+0.2;
        WaypointProx   = 0.2;
        vehicleType     = 2; % 1:int, 2:dint, 3:uav, 4:ugv, 5: h-vtol
        vehicleMode     = 2; % 1:standby, 2:auto
        sensor          = sensorClass.empty;
        QuantifyFlag    = 0;
    end

    properties (Access = private)
        e               = [];
        U               = [];
        Waypoint        = [];
        Waypoints       = [];
        Waypoint_ID     = 1;
        numOfWaypoints  = [];
        numOfSensors    = 0;
        Waypoint_flag   = 0;
    end

    methods

        function obj = vehicleClass(params)
            if nargin == 1
                % UAV Constants
                if isfield(params,'p'),     obj.p       = params.p;     end
                if isfield(params,'v'),     obj.v       = params.v;     end
                if isfield(params,'dt'),    obj.dt      = params.dt;    end
                if isfield(params,'alim'),  obj.alim    = params.alim;  end
                if isfield(params,'vlim'),  obj.vlim    = params.vlim;  end
                if isfield(params,'Kp'),    obj.Kp      = params.Kp;    end
                if isfield(params,'Kd'),    obj.Kd      = params.Kd;    end
                if isfield(params,'WaypointProx')
                    obj.WaypointProx = params.WaypointProx; end
                if isfield(params,'Waypoint')
                    obj.WaypointList = [obj.WaypointList;params.Waypoint];
                    obj.Waypoint = obj.WaypointList(obj.WaypointNum);
                end
            end
            obj.dt                  = 1/obj.Fs; % time step
            obj.e                   = 0;
            obj.U                   = zeros(size(obj.p));
            obj.sensor              = sensorClass();
            obj.sensor.eta          = 0.01;
            obj.sensor(2)           = sensorClass();
            obj.sensor(2).ID        = 2;
            obj.sensor(2).eta       = 0.005;
            obj.sensor(2).alph      = 1;
            obj.numOfSensors        = length(obj.sensor);
        end

        function [obj, flag] = stepDynamics(obj)
            flag = 0;
            if obj.vehicleMode == 2 % auto

                switch obj.vehicleType

                    case 1 % single integator

                    case 2 % double integrator (PD controller)
                        pi = obj.p;
                        vi = obj.v;
                        ei = obj.e;
                        Ui = obj.U;
                        obj.e  = obj.Waypoint - pi;
                        obj.U = obj.Kp.*obj.e + obj.Kd.*(obj.e - ei)./obj.dt;
                        if vecnorm(obj.U,2)> obj.alim
                            obj.U = obj.U./vecnorm(obj.U,2) .* obj.alim;
                        end
                        obj.v = vi + 0.5*obj.dt.*(obj.U + Ui);
                        if vecnorm(obj.v,2)> obj.vlim
                            obj.v = obj.v./vecnorm(obj.v,2) .* obj.vlim;
                        end
                        obj.p   = pi + 0.5*obj.dt.*(obj.v+vi);

                        % -- Take sensor measurements
                        %obj.ws  = ff.wind_measurement(obj.p);
                        %obj.C   = filament.concentration_measurement(obj.p);


                        % -- Reached location
                        flag = norm(obj.p - obj.Waypoint)<obj.WaypointProx;

                end
            end
            if (obj.Waypoint_ID == size(obj.Waypoints,1)) && (obj.Waypoint_flag == 1)
                obj.QuantifyFlag = 0; % set to 0 to not quantify, set to 1 to quantify after waypoint mission
            end
        end

        function obj = step(obj)
            try
                obj.Waypoint = obj.Waypoints(obj.Waypoint_ID,:);
                [obj, obj.Waypoint_flag]     = obj.stepDynamics();

                %After reaching a current waypoint, start going to
                %next waypoint
                if (obj.Waypoint_flag == 1) && (obj.Waypoint_ID < size(obj.Waypoints,1))
                    obj.Waypoint_ID = obj.Waypoint_ID + 1;
                end

                %After passing the first waypoint, do all of this
                %                 if obj.Waypoint_ID > 1
                %
                %                     %Flag being ready to quantify once the final
                %                     %waypoint is reached
                %                     if (obj.Waypoint_ID ==...
                %                             size(obj.Waypoint,1)) && ...
                %                             (obj.Waypoint_flag == 1)
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
            catch
                disp('Vehicle Error:')
            end
        end

        % -- get function
        function waypoint = getWaypoint(obj)
            waypoint = obj.Waypoint;
        end

        % -- get function
        function waypoints = getWaypoints(obj)
            waypoints = obj.Waypoints;
        end

        % -- get function
        function waypoint_id = getWaypoint_ID(obj)
            waypoint_id = obj.Waypoint_ID;
        end

        % -- set function
        %         function obj = setWaypoint(obj,waypoint)
        %             obj.Waypoints = waypoint;
        %             obj.Waypoint_ID = 1;
        %             obj.Waypoint = obj.Waypoints(obj.Waypoint_ID);
        %         end

        % -- set function
        function obj = setWaypoints(obj,waypoints)
            obj.Waypoints = waypoints;
            obj.Waypoint_ID = 1;
            obj.Waypoint = obj.Waypoints(obj.Waypoint_ID,:);
        end

        % -- set function
        function obj = setWaypoint_ID(obj,waypoint_id)
            obj.Waypoint_ID = waypoint_id;
        end

        % -- get Sensor reading functions given input: ff, plume, etc.
        %         function obj = getSensorReading(obj,type,input)
        %             switch type
        %                 case 'point source'
        %                     obj.C   = input.concentration_measurement(obj.p);
        %                 case 'wind'
        %                     obj.ws  = input.wind_measurement(obj.p);
        %             end
        %         end

        function obj = addSensor(obj,sensors)
            if nargin == 2
                len = length(obj.sensor);
                for i=1:length(sensors)
                    params.type = sensors(i);
                    obj.sensor(len+i) = sensorClass(params);
                end
                obj.numOfSensors = length(obj.sensor);
            else
                obj.sensor = sensorClass();
                obj.numOfSensors = 1;
            end
        end

        function pos = getPostion(obj)
            pos = obj.p;
        end



    end

    methods (Access = private)

    end
end
