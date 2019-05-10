
function [angleImg] = GetAngleImage(anglePoint, width, height)
%%% This code is written by BAMI LAB %%%
%%% get angle point to images
count = length(anglePoint);

newImg = zeros(width, height);
for cc= 1:count
    cpoint = anglePoint{cc};
    
    for cp = 1:size(cpoint, 1)
        cpx = cpoint(cp, 2);
        cpy = cpoint(cp, 1);
        newImg(cpx, cpy) = 1;
    end
end

angleImg = logical(newImg);