% -- set simulation params  
simparams.N = 380; 
simparams.modnum = 20;

% -- create param structs for initialization
ff_params               = struct;
ff_params.dim           = 3;
ff_params.uv0           = [1,0];
ff_params.alpha          = 0.3;
ff_params.abG           = [0.005,0.02,3];
plume_params            = struct;
plume_params(1,1).p0    = [80 60 0];   % set initial position 3 dim
plume_params(1,1).Collisions = 'False';
planner_params          = struct;
planner_params.NavStrategy = 'Fluxotaxis';
planner_params.QuantStrategy = 'GDMF';
planner_params.r        = [150,75,2];
planner_params.rd       = [150,75,2];
planner_params.w        = 10;
planner_params.h        = 10; 
planner_params.phi      = 90;
planner_params.phid     = 90;

vehicle_params          = struct;