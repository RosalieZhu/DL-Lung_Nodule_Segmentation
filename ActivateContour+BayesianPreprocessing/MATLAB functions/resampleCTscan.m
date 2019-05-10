% RESAMPLECTSCAN Resample the CT scan.
% This is used for the raw CT lung scans from the LUNA16 dataset,
% and therefore may not be applicable to all other data sources.
% Since the dataset we are interested in is of numeric type "double", we are not
% worrying about other numeric types. You may need to modify this code if you
% want to work with other stuff.
%
% The sampling uses bicubic interpolation.
%
% raw_scan: unprocessed, 3D CT lung scan read from *.mhd files stored as double precision floating numbers.
% current_spacing: the voxel spacing of the raw scan in the form of [x_spacing, y_spacing, z_spacing].
% desired_spacing: the voxel spacing to be expected in the resampled scan.
% resampled_scan: the resampled scan.
%
% April 19, 2019
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University

function resampled_scan = resampleCTscan(raw_scan, current_spacing, desired_spacing)
    % Check the numeric type of the raw scan. Throw an error if it is not "double".
    if ~strcmp(class(raw_scan), 'double')
        error('The input raw scan is not of numeric type "double".');
    end
    
    % The default desired spacing is 1x1x1 mm.
    if nargin < 3
        desired_spacing = [1, 1, 1];
    elseif nargin > 3
        error('Too many input arguments.')
    end
    
    % Check the validity of the current and desired spacing (every element should be greater than 0).
    if ~all(current_spacing > 0)
        error('Not all current spacing are greater than 0.')
    end
    if ~all(desired_spacing > 0)
        error('Not all desired spacing are greater than 0.')
    end
        
    % Calculate the upsampling/downsampling factors based on the current and desired voxel spacing.
    resampling_factors = current_spacing ./ desired_spacing;
    factor_x = resampling_factors(1);
    factor_y = resampling_factors(2);
    factor_z = resampling_factors(3);
    
    % Fill in the correct values for every voxel in the resampled scan.
    resampled_scan = raw_scan(round(1:1./factor_x:size(raw_scan, 1)), round(1:1./factor_y:size(raw_scan, 2)), round(1:1./factor_z:size(raw_scan, 3)));

end