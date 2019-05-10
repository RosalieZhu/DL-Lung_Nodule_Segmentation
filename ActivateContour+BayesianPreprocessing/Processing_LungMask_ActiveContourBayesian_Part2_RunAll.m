% This is a MATLAB script to generate the
% ground truth scans from the LUNA16 3D lung CT
% dataset.
%
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University
% April 21, 2019

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
%save_path = strcat(JUPYTER_DIR, slash, 'Results');
storage_path = '/media/raphael/Raphael Invasion';
save_path = strcat(storage_path, slash, 'Results');
preprocessed_scans_save_path = strcat(save_path, slash, 'Preprocessed Scans');
segmented_scans_save_path = strcat(save_path, slash, 'Segmented Scans');
nodule_masks_save_path = strcat(save_path, slash, 'Nodule Masks');

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
if ~(exist(nodule_masks_save_path, 'dir') == 7)
    mkdir ((nodule_masks_save_path));
end

%% Define the LIDC UID header.
% This is the UID head (all numbers and dots before the last chunk)
% shared by all StudyInstanceUIDs and SeriesInstanceUIDs in the LIDC
% dataset. This implies it is also shared in the LUNA16 dataset.
UID_head = '1.3.6.1.4.1.14519.5.2.1.6279.6001.';

%% Read the LIDC SeriesInstanceUID to StudyInstanceUID mapping file and create a table.
% We will need to extract the SeriesInstanceID to identify the 3D scans
% (either preprocessed or segmented) and extract the corresponding
% StudyInstanceUID to find the corresponding 2D nodule annotation mask
% given by the radiologists.

% NOTE: We found that for long numbers like these UIDs, the MATLAB function
% 'readtable' tends to mess up with the numbers in a non-decodable manner.
% Therefore we can use that function to count the number of rows in the
% *.txt file but not to extract the correct values.

% Open the *.txt file that stored the IDmapping pairs with 'readtable'.
IDtable_bad = readtable(strcat(LIDC_IDmapping_path, slash, 'IDmapping.csv'));
% Read the number or rows and columns of the table.
IDmapping_size = size(IDtable_bad);
% Delete the table because the values are unusable.
clear IDtable_bad;

% Create an empty table of type 'string' to store the UID pairs.
% It should be 1 row shorter than 'IDtable_bad' because it doesn't include
% the header.
IDtable = strings(IDmapping_size(1) - 1, IDmapping_size(2));
% Open the *.txt file that stored the IDmapping pairs.
fileID = fopen(strcat(LIDC_IDmapping_path, slash, 'IDmapping.txt'), 'r');
% Every time 'fgetl(fileID)' is called, a new line from the text is read.
% Ignore the first line because it is the header.
fgetl(fileID);
% Read the second line.
current_line = fgetl(fileID);
% Initialize the line index as 1.
line_idx = 1;
% Repeat throughout the *.txt UID pair file.
while ischar(current_line)
    % Split the current line by the delimiter ','.
    current_pair = split(current_line, ',');
    % Add the StudyInstanceUID and SeriesInstanceUID to the table.
    IDtable{line_idx, 1} = current_pair{1};
    IDtable{line_idx, 2} = current_pair{2};
    % Count up the line index.
    line_idx = line_idx + 1;
    % Read the next line.
    current_line = fgetl(fileID);
end
% Close the *.txt file.
fclose(fileID);

% Clear the duplicates within the table.
IDmapping = unique(IDtable, 'rows');

%% Read the LIDC official nodule annotation files and generate ground truth scans.
% StudyInstanceUIDs are stored in the first column of IDmapping, while
% SeriesInstanceUIDs are stored in the second column of IDmapping.
for row_idx = 1 : size(IDmapping, 1)

    % Since our LUNA16 dataset is a subset of LIDC, not all scans will be
    % found. Thus we create a list to store all SeriesInstanceUIDs that
    % correspond to unsuccessful processing, which are very likely due to
    % the fact that they are not in LUNA16.
    bad_files = [];

    try
        StudyInstanceUID_tail = IDmapping(row_idx, 1);
        SeriesInstanceUID_tail = IDmapping(row_idx, 2);

        % SeriesInstanceUID corresponds to the name of the 3D lung CT scans.
        % Read the size, voxel spacing and origin coordinates of the raw scans.
        SeriesInstanceUID = strcat(UID_head, SeriesInstanceUID_tail);
        [scan_size, scan_voxel_spacing, scan_origin] = read_spacing_from_raw_scans(SeriesInstanceUID, MedicalImageProcessingToolbox_path, luna_dataset_path, slash);

        % Read both the preprocessed and segmented scans.
        %[preprocessed_scan, segmented_scan] = read_preprocessed_and_segmented_scans(SeriesInstanceUID, preprocessed_scans_save_path, segmented_scans_save_path, slash);

        % StudyInstanceUID_tail corresponds to the folder name that contains
        % the 2D nodule masks for that specific scan. Read it.
        [GT_scan_radiologist1, GT_scan_radiologist2, GT_scan_radiologist3, GT_scan_radiologist4] = ...
            apply_radiologist_nodule_mask(StudyInstanceUID_tail, LIDC_annotation_path, scan_size, scan_voxel_spacing, scan_origin, slash);

        % Reshape the resulting ground truth scans to 1x1x1 mm voxel spacing just like in part 1.
        desired_voxel_spacing = [1, 1, 1];
        desired_size = round(scan_voxel_spacing./desired_voxel_spacing.*scan_size);
        GT_scan_radiologist1 = logical(round(imresize3(double(GT_scan_radiologist1), desired_size)));
        GT_scan_radiologist2 = logical(round(imresize3(double(GT_scan_radiologist2), desired_size)));
        GT_scan_radiologist3 = logical(round(imresize3(double(GT_scan_radiologist3), desired_size)));
        GT_scan_radiologist4 = logical(round(imresize3(double(GT_scan_radiologist4), desired_size)));

        % Create file names for the ground truth scans.
        GT_scan_radiologist1_name = strcat(nodule_masks_save_path, slash, SeriesInstanceUID, '_GT1.dcm');
        GT_scan_radiologist2_name = strcat(nodule_masks_save_path, slash, SeriesInstanceUID, '_GT2.dcm');
        GT_scan_radiologist3_name = strcat(nodule_masks_save_path, slash, SeriesInstanceUID, '_GT3.dcm');
        GT_scan_radiologist4_name = strcat(nodule_masks_save_path, slash, SeriesInstanceUID, '_GT4.dcm');

        % Reshape the ground truth scans to [x-dimension, y-dimension, 1,
        % z-dimension]. This is the standard of DICOM. Also convert to int16.
        GT_scan_radiologist1 = reshape(int16(GT_scan_radiologist1), [size(GT_scan_radiologist1, 1), size(GT_scan_radiologist1, 2), 1, size(GT_scan_radiologist1, 3)]);
        GT_scan_radiologist2 = reshape(int16(GT_scan_radiologist2), [size(GT_scan_radiologist2, 1), size(GT_scan_radiologist2, 2), 1, size(GT_scan_radiologist2, 3)]);
        GT_scan_radiologist3 = reshape(int16(GT_scan_radiologist3), [size(GT_scan_radiologist3, 1), size(GT_scan_radiologist3, 2), 1, size(GT_scan_radiologist3, 3)]);
        GT_scan_radiologist4 = reshape(int16(GT_scan_radiologist4), [size(GT_scan_radiologist4, 1), size(GT_scan_radiologist4, 2), 1, size(GT_scan_radiologist4, 3)]);

        % Save the ground truth scans in DICOM format.
        % The '{1}' is just to extract the single-quote strings from
        % double-quote strings. Otherwise the strings won't be treated as
        % directories.
        dicomwrite(GT_scan_radiologist1, GT_scan_radiologist1_name{1});
        dicomwrite(GT_scan_radiologist2, GT_scan_radiologist2_name{1});
        dicomwrite(GT_scan_radiologist3, GT_scan_radiologist3_name{1});
        dicomwrite(GT_scan_radiologist4, GT_scan_radiologist4_name{1});
    catch some_error
        bad_files = [bad_files; SeriesInstanceUID];
        disp(strcat('File cannot be processed. Index number: ', num2str(row_idx)))
    end
end