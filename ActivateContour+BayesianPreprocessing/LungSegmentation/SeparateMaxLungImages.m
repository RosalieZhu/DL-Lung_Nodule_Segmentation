
function [maxLung] = SeparateMaxLungImages(lung)
%%% This code is made by BAMI LAB %%%
% Separate left lung and right lung from lung image.
%   input  : lung binary image
%   output : left lung and right lung image

obj = regionprops(logical(lung));
if isempty(obj)
    maxLung = false(size(lung));
    return;
end

[sortArea, ~]  = sort([obj.Area], 'descend');

mLung = bwareaopen(lung, sortArea(1));
maxLung = mLung;