%% UAV Class
% ----------------------
% by Derek Hollenbeck
% 
% The sensor class contains all the parameters, models, and filters
% associated with the desired sensor type. 
%
% EXAMPLE:
%   ...
% ----------------------
classdef sensorClass
    
    properties 
        % -- DEFAULT params
        ID              = 1;    % 1:wind , 2:concentration, 
        type            = 'point';  % 'point','path-int'
        y               = [];   % sensor output
        Fs              = 20;   % samplerate (for filtering)
        alph            = 0.8;   % LPF parameter
        tau             = 1e-4;   % minimum sensor threshold
        eta             = 0;
    end
    
    properties (Access = private)
       
    end
    
    methods
        
        function obj = sensorClass(params)
            if nargin == 1
                if isfield(params,'ID'),     obj.ID       = params.ID;     end
                if isfield(params,'type'),     obj.type       = params.type;     end
                if isfield(params,'Fs'),    obj.Fs      = params.Fs;    end
                if isfield(params,'alph'),  obj.alph    = params.alph;  end
                if isfield(params,'tau'),  obj.tau    = params.tau;  end
            else
                obj.ID = 1;
                obj.type = 'point';

            end
        end
        
        function [obj, flag] = measure(obj,ff,filament)                       
            flag = [];
        end

        function sensorType = getSensorType(obj)
            sensorType = obj.type;
        end
        
    end
    
    methods (Access = private)
 
    end
end
