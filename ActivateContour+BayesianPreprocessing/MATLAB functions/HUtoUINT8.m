% HUTOUINT8 Converting Hounsfield Unit (Medical Imaging Unit) to UINT8 (integers in [0, 255])
%
% 1. Find and store the min and max values in the original image IMG_HU as a 1-by-2 array in HU_RANGE.
% 2. Convert IMG_HU linearly to its UINT8 counterpart, IMG_UINT8.
% 3. Return IMG_UINT8 and HU_RANGE.
% Note: The image can be 2D planar as well as 3D volumetric.
%
% April 13, 2019.
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University

function [img_UINT8, HU_range] = HUtoUINT8(img_HU)
    if nargin > 1
        error('Too many input arguments');
    end
    
    % Convert the image to double precision first for all following calculations.
    img_HU = double(img_HU);
    
    % Find the min and max values in the original image img_HU and store them as HU_range = [min_HU, max_HU].
    min_HU = min(img_HU(:));
    max_HU = max(img_HU(:));
    HU_range = [min_HU, max_HU];
    
    % Convert img_HU linearly to its UINT8 counterpart, img_UINT8.
    img_UINT8 = img_HU - min(img_HU(:));
    img_UINT8 = uint8(double(255).*img_UINT8./max(img_UINT8(:)));
end