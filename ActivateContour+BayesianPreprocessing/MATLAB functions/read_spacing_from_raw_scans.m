function [scan_size, scan_voxel_spacing, scan_origin] = read_spacing_from_raw_scans(SeriesInstanceUID, MedicalImageProcessingToolbox_path, luna_dataset_path, slash)
    addpath(genpath(MedicalImageProcessingToolbox_path));
    raw_scan_file_name = strcat(luna_dataset_path, slash, SeriesInstanceUID, '.mhd');
    % The resulting string has double quotes and therefore
    % doesn't work as we expected. Thus we need to extract the
    % string in single quotes.
    raw_scan_file_name = raw_scan_file_name{1};
    [V, ~] = read_mhd(raw_scan_file_name);
    scan_size = V.size';
    scan_voxel_spacing = V.spacing';
    scan_origin = V.origin';
end