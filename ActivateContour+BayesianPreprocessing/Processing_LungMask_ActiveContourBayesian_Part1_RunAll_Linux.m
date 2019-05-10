% This is a MATLAB script to automate the process of generating the
% preprocessed scans and lung-segmented scans from the LUNA16 3D lung CT
% dataset. This script will skip explanations, so please go find more
% details in the corresponding *.ipynb notebook file.

% This file includes path modifications to run on Linux PC.

% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University
% April 20, 2019

%% Setup environment and paths.

% Directory of this workspace.
JUPYTER_DIR = pwd;

% Find the frontslash positions to help creating substrings that represent parent directories.
frontslash_position = strfind(JUPYTER_DIR, '/');

% Directory of the MATLAB package "MedicalImageProcessingToolbox".
MedicalImageProcessingToolbox_path = strcat(JUPYTER_DIR, '/MedicalImageProcessingToolbox');
addpath(genpath(MedicalImageProcessingToolbox_path));

% Luna16 Dataset directory.
%luna_dataset_path = strcat(JUPYTER_DIR, '/LUNA16 Sample Scan');
level_up_directory = 3;
luna_dataset_path = strcat(JUPYTER_DIR(1 : frontslash_position(end-level_up_directory)-1), '/Datasets/LUNA16_dataset/scans_copy');
addpath(genpath(luna_dataset_path));

% Directory of my helper functions for image processing.
function_path = strcat(JUPYTER_DIR, '/MATLAB functions');
addpath(genpath(function_path));

% Directory of the ActiveContourBayesian preprocessing package.
ActiveContourBayesian_path = strcat(JUPYTER_DIR, '/LungSegmentation');
addpath(genpath(ActiveContourBayesian_path));

% Directory to read the nodule annotations from.
nodule_annotation_path = strcat(JUPYTER_DIR, '/Nodule annotations');

% Directory to save the results.
save_path = strcat(JUPYTER_DIR, '/Results');
preprocessed_scans_save_path = strcat(save_path, '/Preprocessed Scans');
segmented_scans_save_path = strcat(save_path, '/Segmented Scans');
nodule_masks_save_path = strcat(save_path, '/Nodule Masks');

% Make these directories to save the results if they do not exist.
if ~(exist(save_path, 'dir') == 7)
    mkdir ([save_path]);
end
if ~(exist(preprocessed_scans_save_path, 'dir') == 7)
    mkdir ([preprocessed_scans_save_path])
end
if ~(exist(segmented_scans_save_path, 'dir') == 7)
    mkdir ([segmented_scans_save_path]);
end
if ~(exist(nodule_masks_save_path, 'dir') == 7)
    mkdir ([nodule_masks_save_path]);
end

%% Read the scan file directory.

% Load the dataset file names. The folder name is automatically read by the "dir" function.
cd(luna_dataset_path);
file_names = dir('*.mhd');
cd(JUPYTER_DIR);

%% Iterate over all files, generate the preprocessed and lung-segmented scans and save them in DICOM.
for file_idx = 1 : length(file_names)
    [V, info] = read_mhd(strcat(file_names(file_idx).folder, '/', file_names(file_idx).name));
    raw_scan = V.data;
    current_voxel_spacing = V.spacing';
    
    desired_voxel_spacing = [1, 1, 1];
    desired_size = round(current_voxel_spacing./desired_voxel_spacing.*size(raw_scan));
    resampled_raw_scan = imresize3(raw_scan, desired_size);
    
    threshold = -800;
    thresholded_resampled_raw_scan = resampled_raw_scan;
    thresholded_resampled_raw_scan(resampled_raw_scan <= threshold) = threshold;
    
    window_width = 1500;
    window_center = -600;
    preprocessed_scan = windowing(thresholded_resampled_raw_scan, [window_width, window_center]);
    scan = HUtoUINT8(preprocessed_scan);
    
    num_slices = size(scan, 3);
    THec = 38;
    THiop = 50;
    lungFinalMask = logical(zeros(size(scan)));
    lungFinalContour = cell(num_slices, 1);
    
    bad_slices = [];
    nearest_valid_slice_idx = 1;
    
    for slice_idx = 1 : num_slices
        current_slice = scan(:, :, slice_idx); % load the current slice.
        getLungMask = FindGlobalLung(current_slice); % find the global lung in the current slice.
        getContour = FindContour(getLungMask); % find the coarse lung contour in the current slice.
    
        lungFinalMask(:, :, slice_idx) = getLungMask;
        lungFinalContour{slice_idx, 1} = getContour;
        
        if slice_idx == nearest_valid_slice_idx + 1
            preData = lungFinalMask(:, :, slice_idx - 1); 
            preContour = lungFinalContour{slice_idx - 1};
        else
            preData = lungFinalMask(:, :, slice_idx);
            preContour = lungFinalContour{slice_idx};
        end
        
        try
            anglePoint = FindAngle(getLungMask, getContour); % find concave point in binary lung image
            [erContour, erLung] = ExpandReduceCorrLung(preData, preContour, getContour); % get Expand and Reduce Model (erModel) from n - 1 final Lung
            diffArea = DifferenceArea(erLung, getLungMask); % difference Area of erModel Lung and Current Lung
            angleArea = RemainAngleArea(diffArea, anglePoint); % find angle Points for difference Area
            sharpenArea = FunctionSharpen(current_slice, angleArea, getContour); % sharpness
            ecArea = EllipseAndCircularCorrelation(sharpenArea, THec); % get Ellipse and Circular Correlation
            inoutArea = FunctionInoutLungArea(ecArea, getLungMask, THiop); % get In & Out calculation
            
            addFinalLung = logical(imadd(inoutArea, getLungMask));
            fillFinalLungMask = imfill(addFinalLung, 'hole');
            finalLungMask = fillFinalLungMask;
            finalContour = FindContour(finalLungMask);

            lungFinalMask(:, :, slice_idx) = finalLungMask;
            lungFinalContour{slice_idx} = finalContour;
            nearest_valid_slice_idx = slice_idx;
        
        catch Some_error
            nearest_valid_slice_idx = slice_idx + 1;
            bad_slices = [bad_slices, slice_idx];
        end
    end
    
    connected_components = bwconncomp(lungFinalMask);
    numVoxels = cellfun(@numel, connected_components.PixelIdxList);
    [two_biggest_components, two_biggest_component_indices] = maxk(numVoxels, 2);

    biggest_component = two_biggest_components(1); second_biggest_component = two_biggest_components(2);
    biggest_component_idx = two_biggest_component_indices(1); second_biggest_component_idx = two_biggest_component_indices(2);

    lungCleanMask = zeros(size(lungFinalMask));
    lungCleanMask(connected_components.PixelIdxList{biggest_component_idx}) = 1;
    if (biggest_component./second_biggest_component < 5) && (second_biggest_component > 0.2.*10.^6)
        lungCleanMask(connected_components.PixelIdxList{second_biggest_component_idx}) = 1;
    end
    
    segmented_scan = preprocessed_scan .* lungCleanMask;
    segmented_scan(lungCleanMask == 0) = 45;
    
    preprocessed_scan = int16(preprocessed_scan);
    segmented_scan = int16(segmented_scan);

    preprocessed_scan = reshape(preprocessed_scan, [size(preprocessed_scan, 1), size(preprocessed_scan, 2), 1, size(preprocessed_scan, 3)]);
    segmented_scan = reshape(segmented_scan, [size(segmented_scan, 1), size(segmented_scan, 2), 1, size(segmented_scan, 3)]);
    
    metadata = struct();
    metadata.WindowCenter = window_center;
    metadata.WindowWidth = window_width;
    
    preprocessed_scan_name = [preprocessed_scans_save_path, '/', file_names(file_idx).name(1:end-4), '_preprocessed.dcm'];
    segmented_scan_name = [segmented_scans_save_path, '/', file_names(file_idx).name(1:end-4), '_segmented.dcm'];
    dicomwrite(preprocessed_scan, preprocessed_scan_name, metadata);
    dicomwrite(segmented_scan, segmented_scan_name, metadata);
end
