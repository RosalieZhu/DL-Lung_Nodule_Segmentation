
function [ecArea] = EllipseAndCircularCorrelation(sharpenArea, THec)
%%% This code is written by BAMI LAB %%%
%%% get Ellipse and Circular Correlation
%%% 각 객체의 최장거리를 구한다.
inoutCont = FindContour(sharpenArea);
iocount = size(inoutCont, 1);

width = size(sharpenArea, 1);
height = size(sharpenArea, 2);

%%% 최장거리를 가지고 radius만큼의 타원을 그려준다.
elipseInOut = false(width, height);

for io = 1:iocount
    inoutImg = GetImageFromContour(inoutCont{io}, size(sharpenArea));
    obj = regionprops(logical(inoutImg));
    cpy = obj.Centroid(1);
    cpx = obj.Centroid(2);
    
    mdX = max(inoutCont{io}(:, 1)) - min(inoutCont{io}(:, 1));
    mdY = max(inoutCont{io}(:, 2)) - min(inoutCont{io}(:, 2));
    rrX = ceil(mdX/2);
    rrY = ceil(mdY/2);
    [cl, rl] = meshgrid(1:height, 1:width);
    circlePixels = ((rl-cpx).^2)/rrY.^2 + ((cl-cpy).^2)/rrX.^2 <= 1;
    
    cidx = find(circlePixels == 1);
    ioidx = find(inoutImg == 1 & circlePixels == 1);
    cioPercent = (length(ioidx)/length(cidx))*100;
    
    if cioPercent > THec
        elipseInOut = logical(imadd(elipseInOut, inoutImg));
    end
end

ecArea = elipseInOut;