
function diffArea = DifferenceArea(erLung, nLung)
%%% This code is written by BAMI LAB %%%
%%% Difference Area of erModel Lung and Current Lung

removeArea = imsubtract(erLung, nLung);
removeArea(find(removeArea == -1)) = 0;
removeArea = logical(removeArea);
diffArea = bwareaopen(removeArea, 30);