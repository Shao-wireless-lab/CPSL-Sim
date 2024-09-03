classdef QControllerClass < dynamicprops
    
    properties

    end
    
    properties (Access = private)
        Type = 'N/A';
    end
 
    methods
        function obj = QControllerClass(params)
            if nargin == 1
                if isfield(params,'Type'),  obj.Type    = params.Type;  end
  
                switch obj.Type
                    
                    %Navigation Controller Strategy Params
                    case 'LawnMower'
                        % -- Plane
                        obj.addprop('Zlim');
                        obj.addprop('Width');
                        obj.addprop('Angle');
                        obj.addprop('LatLon_Position');
                        % -- Lawnmower Row Spacing
                        obj.addprop('dZ');
                        
                        % -- Assign Default Paramers
                        obj.Zlim            = [6 14];       %flux plane vertical bounds
                        obj.Width           = 10;           %flux plane horizontal width
                        obj.dZ              = 2;            %flux plane vertical path steps
                        obj.Angle           = pi/2;         %flux plane angle with respect to x-axis (in radians)
                        obj.LatLon_Position = [30 50];      %flux plane location in XY-plane
                        
                        % -- User-Defined Paramers
                        if isfield(params,'Zlim'),      obj.Zlim    = params.Zlim;  end
                        if isfield(params,'Width'),     obj.Width   = params.Width; end
                        if isfield(params,'dZ'),        obj.dZ      = params.dZ;    end
                        if isfield(params,'Angle'),     obj.Angle   = params.Angle; end
                        if isfield(params,'LatLon_Position'),  obj.LatLon_Position   = params.LatLon_Position;  end
                        
                    case 'CVT'
                        % -- Plane
                        obj.addprop('Zlim');
                        obj.addprop('Width');
                        obj.addprop('Angle');
                        obj.addprop('LatLon_Position');
                        % -- CVT
                        obj.addprop('kp');
                        obj.addprop('kpe');
                        obj.addprop('kv');
                        obj.addprop('tau');
                        obj.addprop('winlen');
                        obj.addprop('Xl');
                        obj.addprop('Xg');
                        
                        % -- Assign Default Paramers
                        obj.Zlim            = [6 14];       %flux plane vertical bounds
                        obj.Width           = 10;           %flux plane horizontal width
                        obj.Angle           = pi/2;         %flux plane angle with respect to x-axis (in radians)
                        obj.LatLon_Position = [30 50];      %flux plane location in XY-plane
                        
                        obj.kp                      = 1;
                        obj.kpe                     = 0;
                        obj.kv                      = 0;
                        obj.tau                     = [];   % Delaunay Triangulation Points
                        obj.winlen                  = 100;  % length of samples used for quantification
                        obj.Xl                      = [];   % local plane coordinates
                        obj.Xg                      = [];   % global plane coordinates
                        
                        % -- User-Defined Paramers
                        if isfield(params,'Zlim'),      obj.Zlim    = params.Zlim;  end
                        if isfield(params,'Width'),     obj.Width   = params.Width; end
                        if isfield(params,'Angle'),     obj.Angle   = params.Angle; end
                        if isfield(params,'LatLon_Position'),  obj.LatLon_Position   = params.LatLon_Position;  end
                        
                        if isfield(params,'kp'),      obj.kp    = params.kp;  end
                        if isfield(params,'kpe'),     obj.kpe   = params.kpe; end
                        if isfield(params,'kv'),     obj.kv   = params.kv; end
                        if isfield(params,'tau'),  obj.tau   = params.tau;  end
                        if isfield(params,'winlen'),      obj.winlen   = params.winlen;  end
                        if isfield(params,'Xl'),     obj.Xl   = params.Xl; end
                        if isfield(params,'Xg'),     obj.Xg   = params.Xg; end
                        
                    %Quantification Strategy Params
                    case 'NGI'
                        % -- General
                        obj.addprop('p');
                        obj.addprop('c');
                        obj.addprop('ws');
                        obj.addprop('qm');
                        obj.addprop('Q');
                        % -- NGI Optimization Parameters
                        obj.addprop('sigy');
                        obj.addprop('sigz');
                        obj.addprop('Ty');
                        obj.addprop('Tz');
                        obj.addprop('x_hat');
                        
                        % -- Assign Default Paramers
                        obj.p                       = []; % stored position
                        obj.c                       = []; % stored concentration
                        obj.ws                      = []; % stored ws speed
                        obj.qm                      = []; % measured flux
                        obj.Q                       = []; % estimated source rate
                        
                        obj.sigy                    = []; % estimated dispersion in y
                        obj.sigz                    = []; % estimated dispersion in z
                        obj.Ty                      = []; % estimated scale factor in y
                        obj.Tz                      = []; % estimated scale factor in z
                        obj.x_hat                   = 10; % estimated distance to the source
                        
                        % -- User-Defined Paramers
                        if isfield(params,'p'),     obj.p       = params.p;  end
                        if isfield(params,'c'),     obj.c       = params.c;  end
                        if isfield(params,'ws'),    obj.ws      = params.ws; end
                        if isfield(params,'qm'),    obj.qm      = params.qm; end
                        if isfield(params,'Q'),     obj.Q       = params.Q;  end
                        
                        if isfield(params,'sigy'),  obj.sigy    = params.sigy; end
                        if isfield(params,'sigz'),  obj.sigz    = params.sigz; end
                        if isfield(params,'Ty'),    obj.Ty      = params.Ty;   end
                        if isfield(params,'Tz'),    obj.Tz      = params.Tz;   end
                        if isfield(params,'x_hat'), obj.x_hat   = params.x_hat;end
                        
                    case 'MB'
                        % -- General
                        obj.addprop('p');
                        obj.addprop('c');
                        obj.addprop('ws');
                        obj.addprop('qm');
                        obj.addprop('Q');
                        % -- Kriging Parameters
                        obj.addprop('distTypes');
                        obj.addprop('lag');
                        obj.addprop('alph');
                        
                        % -- Assign Default Paramers
                        obj.p                       = []; % stored position
                        obj.c                       = []; % stored concentration
                        obj.ws                      = []; % stored ws speed
                        obj.qm                      = []; % measured flux
                        obj.Q                       = []; % estimated source rate
                        
                        obj.distTypes               = {'blinear','circular','spherical','pentaspherical',...
                            'exponential','gaussian','whittle','stable','matern'};
                        obj.lag                     = 15;
                        obj.alph                    = 0.7;
                        
                        % -- User-Defined Paramers
                        if isfield(params,'p'),     obj.p       = params.p;  end
                        if isfield(params,'c'),     obj.c       = params.c;  end
                        if isfield(params,'ws'),    obj.ws      = params.ws; end
                        if isfield(params,'qm'),    obj.qm      = params.qm; end
                        if isfield(params,'Q'),     obj.Q       = params.Q;  end
                        
                        if isfield(params,'distTypes'), obj.distTypes   = params.distTypes; end
                        if isfield(params,'lag'),       obj.lag         = params.lag;       end
                        if isfield(params,'alph'),      obj.alph        = params.alph;      end
                        
                    case 'DE-CVT'
                        
                    case 'Mod-NGI'
                        % -- General
                        obj.addprop('p');
                        obj.addprop('c');
                        obj.addprop('ws');
                        obj.addprop('qm');
                        obj.addprop('Q');
                        % -- NGI Optimization Parameters
                        obj.addprop('sigy');
                        obj.addprop('sigz');
                        obj.addprop('Ty');
                        obj.addprop('Tz');
                        obj.addprop('x_hat'); 
                        % -- Kriging Parameters
                        obj.addprop('distTypes');
                        obj.addprop('lag');
                        obj.addprop('alph');
                        
                        % -- Assign Default Paramers
                        obj.p                       = []; % stored position
                        obj.c                       = []; % stored concentration
                        obj.ws                      = []; % stored ws speed
                        obj.qm                      = []; % measured flux
                        obj.Q                       = []; % estimated source rate
                        
                        obj.sigy                    = []; % estimated dispersion in y
                        obj.sigz                    = []; % estimated dispersion in z
                        obj.Ty                      = []; % estimated scale factor in y
                        obj.Tz                      = []; % estimated scale factor in z
                        obj.x_hat                   = 10; % estimated distance to the source
                        
                        obj.distTypes               = {'blinear','circular','spherical','pentaspherical',...
                            'exponential','gaussian','whittle','stable','matern'};
                        obj.lag                     = 15;
                        obj.alph                    = 0.7;
                        
                        % -- User-Defined Paramers
                        if isfield(params,'p'),     obj.p       = params.p;  end
                        if isfield(params,'c'),     obj.c       = params.c;  end
                        if isfield(params,'ws'),    obj.ws      = params.ws; end
                        if isfield(params,'qm'),    obj.qm      = params.qm; end
                        if isfield(params,'Q'),     obj.Q       = params.Q;  end
                        
                        if isfield(params,'sigy'),  obj.sigy    = params.sigy; end
                        if isfield(params,'sigz'),  obj.sigz    = params.sigz; end
                        if isfield(params,'Ty'),    obj.Ty      = params.Ty;   end
                        if isfield(params,'Tz'),    obj.Tz      = params.Tz;   end
                        if isfield(params,'x_hat'), obj.x_hat   = params.x_hat;end
                        
                        if isfield(params,'distTypes'), obj.distTypes   = params.distTypes; end
                        if isfield(params,'lag'),       obj.lag         = params.lag;       end
                        if isfield(params,'alph'),      obj.alph        = params.alph;      end
                end
            end
        end   
    end
    
end