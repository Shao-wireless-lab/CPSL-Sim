% -- set simulation params  
simparams.N = 300; 
simparams.modnum = 20;

% -- create param structs for initialization
ff_params               = struct;
ff_params.dim           = 3;
ff_params.uv0           = [1,0];
ff_params.abG           = [0.005,0.02,1];
% ff_params.abG           = [0,0,0];
%ff_params.uv0           = [1,-1];
%ff_params.abG           = [0.5,0.1,3];
%ff_params.uv0           = [1,0.5];
%ff_params.abG           = [0.4,0.4,5];
%ff_params.alpha          = 0.3;

ff_params.Fs = 20; 

plume_params            = struct;
plume_params(1,1).p0    = [80 60 0];   % set initial position 3 dim
plume_params(1,1).Collisions = 'False';
plume_params.Fs = 20; 

planner_params          = struct;
planner_params.NavStrategy = 'CVT';
planner_params.QuantStrategy = 'Mod-NGI';
planner_params.r        = [40,50,2];
planner_params.rd       = [40,50,2];
planner_params.w        = 30;
planner_params.h        = 10; 
planner_params.phi      = 90;
planner_params.phid     = 90;

vehicle_params.QuantifyFlag = 1; 

planner_params.delx = 2; 
planner_params.dely = 2; 
%planner_params.delz = 2/1.5; 

%planner_params.dataout2 = zeros(200/planner_params.delx,200/planner_params.dely,20/planner_params.delz,simparams.N * ff_params.Fs,2);
%planner_params.dataout2 = zeros(200/planner_params.delx,200/planner_params.dely,20/planner_params.delz,simparams.N * ff_params.Fs,2);
planner_params.xvals = planner_params.delx/2:planner_params.delx:200; 
planner_params.yvals = planner_params.dely/2:planner_params.dely:200;
%planner_params.zvals = planner_params.delz/2:planner_params.delz:20; 
planner_params.zvals = [2,5]; 

vehicle_params          = struct;