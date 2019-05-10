
function [objectContour] = FindContour(object)
%%% This code is written by BAMI LAB %%%
% Find binary's contour
%   input  :
%       lung = binary image
%
%   output :
%       contour = cell array of lung's contour

bw = bwboundaries(object, 'noholes');
obj = regionprops(logical(object));
[~, sidx]  = sort([obj.Area], 'descend');
count = size(bw, 1);

if count == 0
    contour = [];
else
    contour = cell(count,1);
    for i=1:count
        contourLine = fliplr(bw{sidx(i)});
        contourOverlap = contourLine(1, :);
        
        for cc = 2:size(contourLine, 1)
            if sum(contourOverlap(:, 1) == contourLine(cc, 1) & contourOverlap(:, 2) == contourLine(cc, 2)) > 0
                continue;
            else
                contourOverlap(end+1, :) = contourLine(cc, :);
            end
        end
        contour{i} = contourOverlap;
    end
end
objectContour = contour;