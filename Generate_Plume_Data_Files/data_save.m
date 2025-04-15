% Define the folder to save files
save_folder = 'G=5_60_120';

% Create the folder if it doesn't exist
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

% Clear variables to reset for next dataset
clear C_columns ws_columns C ws

% Process DATA_Height_2
% Extract every fourth column
C_columns = DATA_Height_2(:, 4:4:end);

% Extract the rest of the columns
ws_columns = DATA_Height_2;
ws_columns(:, 4:4:end) = [];

% Save C_columns to a new .mat file
C = C_columns;
save(fullfile(save_folder, 'Plume-C-Data-Height-2.mat'), 'C');

% Save ws_columns to a new .mat file
ws = ws_columns;
save(fullfile(save_folder, 'Plume-Wind-Data-Height-2.mat'), 'ws');

% Clear variables to reset for next dataset
clear C_columns ws_columns C ws

% Process DATA_Height_5
% Extract every fourth column
C_columns = DATA_Height_5(:, 4:4:end);

% Extract the rest of the columns
ws_columns = DATA_Height_5;
ws_columns(:, 4:4:end) = [];

% Save C_columns to a new .mat file
C = C_columns;
save(fullfile(save_folder, 'Plume-C-Data-Height-5.mat'), 'C');

% Save ws_columns to a new .mat file
ws = ws_columns;
save(fullfile(save_folder, 'Plume-Wind-Data-Height-5.mat'), 'ws');

% Clear variables to reset for next dataset
clear C_columns ws_columns C ws

disp('Files saved successfully into folder "ProcessedData".');
