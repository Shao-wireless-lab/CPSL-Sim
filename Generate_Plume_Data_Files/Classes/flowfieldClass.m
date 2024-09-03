%% Flowfield Class
% ----------------------
% by Derek Hollenbeck
% 
% ...
%
% EXAMPLE:
%   ...
% ----------------------

classdef flowfieldClass
    
    properties 
        % default params and variables
        K           = [1000,1000];  % diffusivity 
        n           = 20;           % rows of flowfield
        m           = 20;           % columns of flowfield
        l           = 10;            % height of flowfield
        dim         = 2;            % dimension of flowfield
        uv          = [];           % flow field
        %abG         = [0.01,0.01,5];% colored noise params
        abG         = [0.5,0.1,3];% colored noise params
        uv0         = [1,0];        % initial/mean wind direction 
        tf_num      = [];           % transfer function numerator
        tf_den      = [];           % transfer function denominator
        xmin        = 0;            % min range in x-axis
        xmax        = 200;          % max range in x-axis
        ymin        = 0;            % min range in y-axis
        ymax        = 200;          % max range in y-axis
        zmin        = 0;            % min range in z-axis
        zmax        = 20;           % max range in z-axis     
        Fs          = 20;           % sampling frequency
        cn          = [];           % colored noise
        wn          = [];           % white noise
        zf          = [];
        dx          = [];
        dy          = [];
        dz          = [];
        dt          = [];
        alph        = 0.3;           %Power law scaling exponent
    end
    
    properties (Access = private)
        % properties exclusive to environment class
        currentTime = 0;
        uv_temp     = [];
    end
    
    methods
        
        function obj = flowfieldClass(params)
            if nargin == 1
                if isfield(params,'K'),     obj.K       = params.K;     end
                if isfield(params,'n'),     obj.n       = params.n;     end
                if isfield(params,'m'),     obj.m       = params.m;     end
                if isfield(params,'l'),     obj.l       = params.l;     end
                if isfield(params,'dim'),   obj.dim     = params.dim;   end
                if isfield(params,'abG'),   obj.abG     = params.abG;   end
                if isfield(params,'uv0'),   obj.uv0     = params.uv0;   end
                if isfield(params,'xmin'),  obj.xmin    = params.xmin;  end
                if isfield(params,'xmax'),  obj.xmax    = params.xmax;  end
                if isfield(params,'ymin'),  obj.ymin    = params.ymin;  end
                if isfield(params,'ymax'),  obj.ymax    = params.ymax;  end
                if isfield(params,'zmin'),  obj.zmin    = params.zmin;  end
                if isfield(params,'zmax'),  obj.zmax    = params.zmax;  end
                if isfield(params,'Fs'),    obj.Fs      = params.Fs;    end
                if isfield(params,'alph'),  obj.alph    = params.alph;  end
            end
            obj.dx = (obj.xmax-obj.xmin)/obj.n;% change in x-axis
            obj.dy = (obj.ymax-obj.ymin)/obj.m;% change in y-axis
            obj.dz = (obj.zmax-obj.zmin)/obj.l;% change in z-axis
            obj.dt = 1/obj.Fs;         % time step

            
            
            N = [0,0,obj.abG(1)*obj.abG(3)];
            D = [1,obj.abG(2),obj.abG(1)];
            % -- Control toolbox Installed
            %dsys = c2d(tf(N,D),obj.dt);
            %obj.tf_num = cell2mat(dsys.Numerator);
            %obj.tf_den = cell2mat(dsys.Denominator);
            % -- No Control Toolbox installed
            T = 1/obj.Fs;
            a = obj.abG(1); b = obj.abG(2); G = obj.abG(3);
            n2 = a*G; n1 = 2*n2; n0 = n2;
            d2 = (4/T^2 + 2*b/T + a); d1 = (-8/T^2 + 2*a); d0 = (4/T^2 - 2*b/T + a);
            obj.tf_num = [n2/d2,n1/d2,n0/d2];
            obj.tf_den = [1,d1/d2,d0/d2];
            
            obj.uv_temp = zeros(obj.n,obj.m,obj.dim);
            obj.uv = zeros(obj.n,obj.m,obj.dim);
            
            if obj.dim == 3
                obj.uv = zeros(obj.l,obj.n,obj.m,obj.dim);
            end
        end
        
        function [obj, flag] = step(obj)
            
            if obj.dim == 2
                obj.uv_temp = obj.uv;
                [obj,flag] = step2D(obj);
                obj.uv = obj.uv_temp;
                
            elseif obj.dim == 3
                obj.uv_temp = squeeze(obj.uv(1,:,:,1:2));
                [obj,flag] = step2D(obj);
                obj.uv_temp(:,:,3) = 0;
                obj.uv(1,:,:,:) = obj.uv_temp;
                z = linspace(obj.zmin,obj.zmax,obj.l);
                for i = 2:obj.l
                    obj.uv(i,:,:,:) = obj.uv_temp.*(z(i)/2).^obj.alph...
                                      + 0.1*rand(obj.n,obj.m,3);
                end
                
            else
                disp('Error: Invalid Dimension Value')
            end
        end
        
        function ws = wind_measurement(obj,sensor)
            % sensor is a matrix
            % each row vector is (x, y, [z])
            
            L     = size(sensor, 1); % number of sensors 
            k = zeros(L, obj.dim);
            
            if obj.dim == 2
                k(:,:) = round(sensor ./ ([obj.dx obj.dy])); % discrete position
            elseif obj.dim == 3
                k(:,:) = round(sensor ./ ([obj.dx obj.dy obj.dz])); % discrete position
            end
            
            k(k(:, :) < 1) = 1; % sets lower-bound
            k(k(:, 1) > obj.n, 1) = obj.n; % sets upper-bound
            k(k(:, 2) > obj.m, 2) = obj.m; % sets upper-bound
            
            if obj.dim == 3
                k(k(:, 3) > obj.l, 3) = obj.l; % sets upper-bound
            end
            
            k = fliplr(k); % [x, y, (z)] to [(z), y, x], this makes it easier to index the matrix
            
            if obj.dim == 2
                indeces = [...
                    sub2ind(size(obj.uv), k(:, 1), k(:, 2),     ones(size(k, 1), 1)),... % u component at point x
                    sub2ind(size(obj.uv), k(:, 1), k(:, 2), 2 * ones(size(k, 1), 1)) ... % v component at point y
                    ];
            elseif obj.dim == 3
                % indeces = [...
                %     sub2ind(size(obj.uv), k(:, 1), k(:, 2),k(:, 3),     ones(size(k, 1), 1)),... % u component at point x
                %     sub2ind(size(obj.uv), k(:, 1), k(:, 2),k(:, 3), 2 * ones(size(k, 1), 1)), ... % v component at point y
                %     sub2ind(size(obj.uv), k(:, 1), k(:, 2),k(:, 3), 3 * ones(size(k, 1), 1)) ... % w component at point z
                %     ];
                indeces = [...
                    sub2ind(size(obj.uv), k(:, 1), k(:, 3),k(:, 2),     ones(size(k, 1), 1)),... % u component at point x
                    sub2ind(size(obj.uv), k(:, 1), k(:, 3),k(:, 2), 2 * ones(size(k, 1), 1)), ... % v component at point y
                    sub2ind(size(obj.uv), k(:, 1), k(:, 3),k(:, 2), 3 * ones(size(k, 1), 1)) ... % w component at point z
                    ];
            end
            
            ws = obj.uv(indeces);
        end
    end

    methods (Static)
        function showCurrentTime(obj)
            disp(getCurrentTime(obj))
        end
        function updateCurrentTime(obj,time)
            obj = setCurrentTime(obj,time);
        end
    end

    methods (Access = private)
        function [obj,flag] = step2D(obj)
            flag = 0;
            if isempty(obj.cn) % if there are no more colored noise, generate some
                l = 10000; % arbitrary number choice
                obj.wn = randn(4, 2, l); % white noise; 2 is for 2D

                % generate l sets of colored noise vectors and store it
                % the filter's state "zf" will provide memory of the previous noise, for the next time we generate more noise
                if ~isfield(obj, 'zf')
                    obj.zf = [];
%                     obj.Gzf = [];
                end
                %Gegenbauer Filter
%                 [obj.Gn, obj.Gzf] = filter(tf_Gegenbauer, 1, obj.wn, obj.Gzf, 3);
                %Colored Noise Filter
                [obj.cn, obj.zf] = filter(obj.tf_num, obj.tf_den, obj.wn, obj.zf, 3); % 3 refers to the third dimension being time
                flag = 1;
            end

            % grab colored noise corner values
            cn_corner_vectors = obj.cn(:, :, 1); % pop one set of colored noise
            obj.cn(:, :, 1) = [];
      
            % update the boundaries
            t = (cn_corner_vectors(1, :) - cn_corner_vectors(2, :)) ./ (1 - obj.m) .* ((1:obj.m).' - 1) + cn_corner_vectors(1, :) + obj.uv0;
            l = (cn_corner_vectors(1, :) - cn_corner_vectors(3, :)) ./ (1 - obj.n) .* ((1:obj.n).' - 1) + cn_corner_vectors(1, :) + obj.uv0;
            r = (cn_corner_vectors(2, :) - cn_corner_vectors(4, :)) ./ (1 - obj.n) .* ((1:obj.n).' - 1) + cn_corner_vectors(2, :) + obj.uv0;
            b = (cn_corner_vectors(3, :) - cn_corner_vectors(4, :)) ./ (1 - obj.m) .* ((1:obj.m).' - 1) + cn_corner_vectors(3, :) + obj.uv0;
            obj.uv_temp(1, :, :) = t;
            obj.uv_temp(:, 1, :) = l;
            obj.uv_temp(:, obj.m, :) = r;
            obj.uv_temp(obj.n, :, :) = b;

            % adding mask for buildings
            %     if isfield(ff_props, 'mask')
            %         ff = ff.*mask;
            %     end

            % first, convert n x m flowfield matrix to an nm long vector
            u_n = reshape(obj.uv_temp(:, :, 1), [obj.n * obj.m 1]);
            v_n = reshape(obj.uv_temp(:, :, 2), [obj.n * obj.m 1]);

            % use this to make vectors of the same element
            e = ones(obj.n*obj.m, 1);

            % constants from Gonca ÇELIKTEN 1 and Emine Nesligul AKSAN 2 paper
            rx_1 = obj.dt / 4 / obj.dx;
            ry_1 = obj.dt / 4 / obj.dy;
            rx_2 = obj.dt / 2 / obj.dx / obj.dx * obj.K(1) * 0.5;
            ry_2 = obj.dt / 2 / obj.dy / obj.dy * obj.K(2) * 0.5;

            % the first argument to spdiags is row vectors representing the elements along the diagonals
            % the second argument is a vector representing which diagonal the diagonals will be placed on
            % in this case, the higher indexed terms are below the main diagonal, and the lower indexed terms are above the main diagonal
            % This is because we're transposing the matrix afterwards.
            % we transpose the matrix because spdiags line up elements of the same index vertically, whereas we want them to line up horizontally

            %                      ( i-1, j term    )  ( i, j term      )   ( i+1, j term  )
            A = spdiags([-rx_1 * u_n - rx_2  (1 + 2 * rx_2) * e   rx_1 * u_n - rx_2], [obj.n 0 -obj.n], obj.n*obj.m, obj.n*obj.m).';
            B = spdiags([ ry_1 * v_n + ry_2  (1 - 2 * ry_2) * e  -ry_1 * v_n + ry_2], [obj.n 0 -obj.n], obj.n*obj.m, obj.n*obj.m).';

            % insert rows from the identity matrix the coefficient matrix to not affect boundary conditions
            A = obj.boundary_filter(A, obj.n, obj.m);
            B = obj.boundary_filter(B, obj.n, obj.m);

            u_n5 = A\(B * u_n); % solve for u_n + 1/2
            v_n5 = A\(B * v_n); % solve for v_n + 1/2

            u_n5_l = u_n5; % linearization of u_n5
            v_n5_l = v_n5; % linearization of v_n5

            %                      ( i, j-1 term       )  ( i, j term      )   ( i, j+1 term     )
            A = spdiags([-ry_1 * v_n5_l - ry_2  (1 + 2 * ry_2) * e   ry_1 * v_n5_l - ry_2], [1 0 -1], obj.n*obj.m, obj.n*obj.m).';
            B = spdiags([ rx_1 * u_n5_l + rx_2  (1 - 2 * rx_2) * e  -rx_1 * u_n5_l + rx_2], [1 0 -1], obj.n*obj.m, obj.n*obj.m).';

            % put identity into the matrix to not affect boundary conditions
            A = obj.boundary_filter(A, obj.n, obj.m);
            B = obj.boundary_filter(B, obj.n, obj.m);

            u_n1 = A\(B * u_n5);
            v_n1 = A\(B * v_n5);

            % convert nm long vector back into n x m flowfield matrix
            obj.uv_temp(:, :, 1) = reshape(u_n1, [obj.n obj.m]);
            obj.uv_temp(:, :, 2) = reshape(v_n1, [obj.n obj.m]);

%             % apply mask again
%             if isfield(obj, 'mask')
%                 mask   = obj.mask;
%                 obj.uv_temp = obj.uv_temp.*mask;
%             end

        end

        function [A] = boundary_filter(obj,A, n, m)
            % Puts identity rows into the A matrix as to leave the boundaries alone
            I = eye(n*m, 'like', A);
            % left boundary
            A(1:n, :) = I(1:n, :);
            % right boundary
            A(n*(m-1):n*m, :) = I(n*(m-1):n*m, :);
            % top boundary
            A(1:n:n*m, :) = I(1:n:n*m, :);
            % bottom boundary
            A(n:n:n*m, :) = I(n:n:n*m, :);
        end
        function [time]=getCurrentTime(obj)
            time = obj.currentTime;
        end
        function [obj] = setCurrentTime(obj,time)
            obj.currentTime = time;
        end
    end
    
end