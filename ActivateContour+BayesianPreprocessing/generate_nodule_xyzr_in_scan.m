% This is a MATLAB script to take the LUNA16 nodule annotation file that
% specified the (x, y, z, r) location in world coordinates and generate a
% corresponding (x, y, z, r) location in the scan coordinates. Since the
% preprocessed, lung-segmented and ground truth scans share the exact same
% coordinate system, the resulting information can be used by all these
% scans listed.
%
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University
% April 25, 2019

%% Define the 'slash' depending on the OS.
% If Windows -> '\'
% If Linux or Mac OS -> '/'
slash = '/';

%% Setup environment and paths.
% Directory of this workspace.
JUPYTER_DIR = pwd;

% Find the front/back slash positions to help creating substrings that represent parent directories.
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

% Directory to read the LUNA16 nodule annotations from.
level_up_directory = 3;
nodule_annotation_path = strcat(JUPYTER_DIR(1 : slash_position(end-level_up_directory)-1), slash, 'Datasets', slash, 'LUNA_annotation');
clear level_up_directory;
addpath(genpath(nodule_annotation_path));

% Directory to save the results.
save_path = strcat(JUPYTER_DIR);

%% Preview the LUNA16 nodule annotation file
annotation_file = strcat(nodule_annotation_path, slash, 'annotations.csv');
import_options = detectImportOptions(annotation_file);
% Uncomment the following line to see the preview
preview(annotation_file, import_options)

%% Load the LUNA16 nodule annotation file
% Read the annotation.csv file as a table.
annotation = readtable(annotation_file);

%% Load the metadata of one CT scan at a time from the LUNA16 dataset and update the (x, y, z, r) for that scan.
% Load all SeriesInstanceUIDs from the *.mhd files of the LUNA16 dataset.
cd(luna_dataset_path);
file_names = dir('*.mhd');
cd(JUPYTER_DIR);

% Use a list of strings to store the SeriesInstanceUIDs.
SeriesInstanceUID = strings(length(file_names), 1);
for file_idx = 1 : length(file_names)
    SeriesInstanceUID{file_idx, 1} = file_names(file_idx).name(1 : end-4);
end

% Duplicate the table 'annotation'.
annotation_in_scan = annotation;

% Use a pointer to point at the row to update.
row_to_update = 1;

% Iterate through all LUNA16 files, extract their metadata and update the
% (x, y, z, r) information of the nodules in 'annotation_in_scan' based on
% their actual coordinates in the scans.
for UID_idx = 1 : length(SeriesInstanceUID)
    [~, ~, raw_scan_origin] = read_spacing_from_raw_scans(SeriesInstanceUID(UID_idx), MedicalImageProcessingToolbox_path, luna_dataset_path, slash);

    % Load all annotation corresponding to the current file.
    % Create a matrix for the (x, y, z, r) coordinates of the nodules in this scan.
    nodule_xyzr = [];
    % Append the nodule information for every nodule identified in this scan
    % (if the seriesuid in the annotation file matches the scan's file name).
    for row_num = 1 : length(annotation.seriesuid)
        if all(SeriesInstanceUID{UID_idx} == annotation{row_num,1}{1})
            nodule_xyzr = [nodule_xyzr; [annotation{row_num, 2:5}]];
        end
    end

    % The scan voxel spacing (in the three types of scans where we care
    % about the in-scan nodule coordinates), the voxel spacings are all
    % 1x1x1 mm.
    scan_voxel_spacing = [1, 1, 1];
    % Find the corresponding coordinates in the scan.
    nodule_xyzr_in_scan = zeros(size(nodule_xyzr));
    % The x-, y-, and z- coordinates in the scan matrix can be calculated from
    % the world coordinates (the coordinate system used by radiologists when
    % annotating the nodules).
    % The nodule radii shall be rescaled depending on the voxel spacing.
    for nodule_idx = 1 : size(nodule_xyzr, 1)
        nodule_xyzr_in_scan(nodule_idx, 1:3) = WORLDtoVOXEL(nodule_xyzr(nodule_idx, 1:3), scan_voxel_spacing, raw_scan_origin);
        nodule_xyzr_in_scan(nodule_idx, 4) = nodule_xyzr(nodule_idx, 4) .* sqrt(sum(scan_voxel_spacing.^2))./sqrt(3);
    end
    
    % Update 'annotation_in_scan' with 'nodule_xyzr_in_scan'.
    if length(nodule_xyzr_in_scan) > 0
        annotation_in_scan{row_to_update : row_to_update+size(nodule_xyzr,1)-1, 2:5} = nodule_xyzr_in_scan(:, 1:4);
    end
    % Count up 'row_to_update'.
    row_to_update = row_to_update + size(nodule_xyzr, 1);
end

%% Save the result.
% Round the numerical values.
annotation_in_scan{:, 2:4} = round(annotation_in_scan{:, 2:4});
writetable(annotation_in_scan, 'annotation_in_scan.csv');