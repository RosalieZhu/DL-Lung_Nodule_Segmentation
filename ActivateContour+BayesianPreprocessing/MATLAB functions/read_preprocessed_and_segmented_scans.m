function [preprocessed_scan, segmented_scan, preprocessed_scan_info, segmented_scan_info] = read_preprocessed_and_segmented_scans(SeriesInstanceUID, preprocessed_scans_save_path, segmented_scans_save_path, slash)
    preprocessed_scan_name = strcat(preprocessed_scans_save_path, slash, SeriesInstanceUID, '_preprocessed.dcm');
    segmented_scan_name = strcat(segmented_scans_save_path, slash, SeriesInstanceUID, '_segmented.dcm');

    % Read the DICOM scans.
    preprocessed_scan = dicomread(preprocessed_scan_name);
    segmented_scan = dicomread(segmented_scan_name);

    % Read the DICOM scan metadata.
    preprocessed_scan_info = dicominfo(preprocessed_scan_name);
    segmented_scan_info = dicominfo(segmented_scan_name);

    % Reshape these files into 3D (x, y, z).
    preprocessed_scan = reshape(preprocessed_scan, [size(preprocessed_scan,1), size(preprocessed_scan,2), size(preprocessed_scan,4)]);
    segmented_scan = reshape(segmented_scan, [size(segmented_scan,1), size(segmented_scan,2), size(segmented_scan,4)]);
end
