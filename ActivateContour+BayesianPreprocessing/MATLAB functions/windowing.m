% WINDOWING Perform a windowing process with specified rescaling window.
% This is a common practice for contrast enhancement in CT imaging
% that utilize a simple intensity transform technique in image processing.
% Note that this is not an invertible transform.
%
% input_img: input image.
% window: [window_width, window_center] specifying the intensity transform window.
% output_img: output image.
% 
% Note: The image can be 2D planar as well as 3D volumetric.
%
% April 13, 2019.
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University

function output_img = windowing(input_img, window)
    % Convert the image to double precision first for all following calculations.
    input_img = double(input_img);

    if nargin == 1
        error('Please provide a window in the form is [window_width, window_center]');
    elif nargin > 2
        error('Too many input arguments');
    end
    
    % Calculate the lower and upper limits for the intensity values.
    width = window(1);
    center = window(2);
    low = center - round(width/2);
    high = center + round(width/2);
    
    % Apply the windowing.
    output_img = input_img;
    output_img(input_img < low) = low;
    output_img(input_img > high) = high;
end