classdef terrainClass

    properties
        % -- DEFAULT PARAMS AND VARIABLES
        dir         = 'Terrain/';
        fn          = 'Metec_Large_Obst';
        surface     = [];
        Nxyz        = [];
        area        = [];
    end

    properties (Access = private)
        % -- DEFAULT PRIVATE PARAMS AND VARIABLE
    end

    methods
        function obj = terrainClass(params)
            if nargin == 1
                if isfield(params,'dir'),   obj.dir     = params.dir;   end
                if isfield(params,'fn'),    obj.fn       = params.fn;   end
            end
            
            obj.surface = load(sprintf('%s%s.mat',obj.dir,obj.fn)); %open terrain file
            obj.surface = eval(sprintf('obj.surface.%s',obj.fn));
            
            [Nx,Ny,Nz] = surfnorm(obj.surface.XData,obj.surface.YData,obj.surface.ZData);
            obj.Nxyz = {Nx Ny Nz};

            % -- Define equipment areas here
            % obj.area = struct;
            % obj.area(1).r = [];
            % obj.area(1).
    
            % -- Define
        end
        
        function collided = collision_measurement(obj,sensor)
            % Note sensor here is the position of the object (i.e. filament) 
            % in the absence of the surface.
            
            %Interpolate to find Z value on surface with same X,Y values
            Zsurf = interp2(obj.surface.XData,...
                            obj.surface.YData,...
                            obj.surface.ZData,...
                            sensor(:,1),sensor(:,2));
            
            %Difference between height of point and point on surface
            
            dZ(sensor(:,3) <= Zsurf) = 1; % If on or under surface, then collision (1)
            dZ(sensor(:,3) > Zsurf) = 0; % If abocve surface, then no collision (0)
            
            collided = dZ';
        end

        function obj = clip2window(obj,params)
            xmin = params(1);
            xmax = params(2);
            ymin = params(3);
            ymax = params(4);

            tempx = obj.surface.XData(1,:);
            tempy = obj.surface.YData(:,1);

            k(1,1) = find(tempx<=xmin,1,"last");
            k(1,2) = find(tempx>=xmax,1,"first");
            k(2,1) = find(tempy>=ymin,1,"first");
            k(2,2) = find(tempy<=ymax,1,"last");

            obj.surface.XData = obj.surface.XData(k(2,1):k(2,2),k(1,1):k(1,2));
            obj.surface.YData = obj.surface.YData(k(2,1):k(2,2),k(1,1):k(1,2));
            obj.surface.ZData = obj.surface.ZData(k(2,1):k(2,2),k(1,1):k(1,2));
            
            [Nx,Ny,Nz] = surfnorm(obj.surface.XData,obj.surface.YData,obj.surface.ZData);
            obj.Nxyz = {Nx Ny Nz};

            obj = obj.zeroHeight();

        end

        function obj = zeroHeight(obj)
            zmin = min(min(obj.surface.ZData));
            obj.surface.ZData = obj.surface.ZData - zmin;
        end
        

    end

    methods (Access = private)

    end

end