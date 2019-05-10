function [GT_scan_radiologist1, GT_scan_radiologist2, GT_scan_radiologist3, GT_scan_radiologist4] = ...
    apply_radiologist_nodule_mask(StudyInstanceUID_tail, LIDC_annotation_path, scan_size, scan_voxel_spacing, scan_origin, slash)
    % Read the *.txt annotation file as a string.
    fileID = fopen(strcat(LIDC_annotation_path, slash, 'gts', slash, StudyInstanceUID_tail, slash, 'slice_correspondences.txt'), 'r');
    formatSpec = '%s';
    annotation_string = fscanf(fileID, formatSpec);
    fclose(fileID);
    % Extract which slice each ground truth belongs to using regular
    % expressions.
    regular_expression = '-zpos:[-]*[0-9]+.[0-9]+';
    [start_idx, end_idx] = regexp(annotation_string, regular_expression);
    % Create a list representing the slices with nodules
    slice_idx_with_nodule = strings(1, length(start_idx));
    for idx = 1 : length(start_idx)
        slice_idx_with_nodule(idx) = annotation_string(start_idx(idx) + 6 : end_idx(idx));
    end
    % Convert the list of strings to list of double precision numbers.
    slice_idx_with_nodule = double(slice_idx_with_nodule);
    
    % Since these annotations were done on the raw scans, we would need to 
    % find the corresponding slice indices in the raw scans. To do that we
    % need the voxel spacing and scan origin information extracted from the
    % metadata.
    slice_idx_with_nodule_in_scan = zeros(size(slice_idx_with_nodule));
    for idx = 1 : length(slice_idx_with_nodule)
        slice_idx_with_nodule_in_scan(idx) = WORLDtoVOXEL(slice_idx_with_nodule(idx), scan_voxel_spacing(3), scan_origin(3));
    end
    % Round the result since a slice number has to be an integers.
    % Theoretically they should already be integers before the rounding.
    slice_idx_with_nodule_in_scan = uint8(slice_idx_with_nodule_in_scan);
    
    % Genereate four ground truth matrices each corresponding to the
    % annotation of a radiologist. These matrices shall be binary: 1 for
    % nodule, 0 for non-nodule.
    GT_scan_radiologist1 = logical(zeros(scan_size));
    GT_scan_radiologist2 = logical(zeros(scan_size));
    GT_scan_radiologist3 = logical(zeros(scan_size));
    GT_scan_radiologist4 = logical(zeros(scan_size));
    
    % Read the ground truth nodule annotation masks.
    for idx = 1 : length(slice_idx_with_nodule_in_scan)
        folder_name = strcat(LIDC_annotation_path, slash, 'gts', slash, StudyInstanceUID_tail, slash, 'slice', num2str(idx));
        GT_slice_radiologist1 = imread(strcat(folder_name, slash, 'GT_id1.tif'));
        GT_slice_radiologist2 = imread(strcat(folder_name, slash, 'GT_id2.tif'));
        GT_slice_radiologist3 = imread(strcat(folder_name, slash, 'GT_id3.tif'));
        GT_slice_radiologist4 = imread(strcat(folder_name, slash, 'GT_id4.tif'));
        
        % Fill the ground truth slices into the ground truth scans.
        % Remember to transpose them to swap the x- and y- to match the
        % preprocessed and segmented scans from part 1.
        slice_idx = slice_idx_with_nodule_in_scan(idx);
        GT_scan_radiologist1(:, :, slice_idx) = GT_slice_radiologist1';
        GT_scan_radiologist2(:, :, slice_idx) = GT_slice_radiologist2';
        GT_scan_radiologist3(:, :, slice_idx) = GT_slice_radiologist3';
        GT_scan_radiologist4(:, :, slice_idx) = GT_slice_radiologist4';
    end
end