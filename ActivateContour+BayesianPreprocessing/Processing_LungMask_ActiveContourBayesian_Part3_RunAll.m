% This is a MATLAB script to generate the
% LUNA16 official lung-segmented scans from the LUNA16 3D lung CT
% dataset.
%
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University
% May 5, 2019

%% Define the 'slash' depending on the OS.
% If Windows -> '\'
% If Linux or Mac OS -> '/'
slash = '/';

%% Setup environment and paths.
% Directory of this workspace.
JUPYTER_DIR = pwd;

% Find the slash positions to help creating substrings that represent parent directories.
slash_position = strfind(JUPYTER_DIR, slash);

% Directory of the MATLAB package "MedicalImageProcessingToolbox".
MedicalImageProcessingToolbox_path = strcat(JUPYTER_DIR, slash, 'MedicalImageProcessingToolbox');
addpath(genpath(MedicalImageProcessingToolbox_path));

% Luna16 Dataset directory.
level_up_directory = 3;
luna_dataset_path = strcat(JUPYTER_DIR(1 : slash_position(end-level_up_directory)-1), slash, 'Datasets', slash, 'LUNA16_dataset', slash, 'scans_all');
clear level_up_directory;
addpath(genpath(luna_dataset_path));

% Directory of my helper functions for image processing.
function_path = strcat(JUPYTER_DIR, slash, 'MATLAB functions');
addpath(genpath(function_path));

% Directory of the LIDC-IDRI official annotation files.
level_up_directory = 3;
LIDC_annotation_path = strcat(JUPYTER_DIR(1 : slash_position(end-level_up_directory)-1), slash, 'Datasets', slash, 'LIDC_annotation');
clear level_up_directory;
addpath(genpath(LIDC_annotation_path));

level_up_directory = 3;
LIDC_IDmapping_path = strcat(JUPYTER_DIR(1 : slash_position(end-level_up_directory)-1), slash, 'Datasets', slash, 'LIDC_IDmapping');
clear level_up_directory;
addpath(genpath(LIDC_IDmapping_path));

% Directory to save the results.
save_path = strcat(JUPYTER_DIR, slash, 'Results');
%storage_path = '/media/raphael/Raphael Invasion';
%save_path = strcat(storage_path, slash, 'Results');
preprocessed_scans_save_path = strcat(save_path, slash, 'Preprocessed Scans');
segmented_scans_save_path = strcat(save_path, slash, 'Segmented Scans');
LUNA_official_segmented_scans_save_path = strcat(save_path, slash, 'LUNA16 Official Segmented Scans');

% Make these directories to save the results if they do not exist.
if ~(exist(save_path, 'dir') == 7)
    mkdir ((save_path));
end
if ~(exist(preprocessed_scans_save_path, 'dir') == 7)
    mkdir ((preprocessed_scans_save_path))
end
if ~(exist(segmented_scans_save_path, 'dir') == 7)
    mkdir ((segmented_scans_save_path));
end
if ~(exist(LUNA_official_segmented_scans_save_path, 'dir') == 7)
    mkdir ((LUNA_official_segmented_scans_save_path));
end

%% Define the LIDC UID header.
% This is the UID head (all numbers and dots before the last chunk)
% shared by all StudyInstanceUIDs and SeriesInstanceUIDs in the LIDC
% dataset. This implies it is also shared in the LUNA16 dataset.
UID_head = '1.3.6.1.4.1.14519.5.2.1.6279.6001.';

end