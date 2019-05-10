
function inoutArea = FunctionInoutLungArea(ecArea, getLung, THiop)
%%% This code is written by BAMI LAB %%%
% get In & Out calculation 

checkArea = ecArea;
obj = regionprops(logical(checkArea));
[~, sortidx]  = sort([obj.Area], 'descend');

width = size(checkArea, 1);
height = size(checkArea, 2);
lengthObj = length(obj);
objLung = checkArea;
inoutAreaCandidate = checkArea;

for lo = 1:lengthObj
    ai = sortidx(lo);
    areaIdx = obj(ai).Area;
    objSeparateLung = bwareaopen(objLung, areaIdx);
    objLung = xor(objLung, objSeparateLung);    
    expLung = imdilate(objSeparateLung, strel('disk', 2));
    subLung = imsubtract(expLung, objSeparateLung);
    
    inCount = 0; % inside lung
    outCount = 0; % outside lung
    totalCount = 0; % total count
    
    for ww = 1:width
        for hh = 1:height
            if subLung(ww, hh) == 1                
                if getLung(ww, hh) == 1
                    inCount = inCount + 1;
                else
                    outCount = outCount + 1;
                end
                totalCount = totalCount + 1;
            else
                continue;
            end
        end
    end
    
    cx = obj(ai).Centroid(1);
    cy = obj(ai).Centroid(2);
    %     insidePercent = inCount / totalCount * 100;
    outsidePercent = outCount / totalCount * 100;
    if isnan(outsidePercent)
        outsidePercent = 100;
    end
    outData = [cx, cy, outsidePercent];
    
    if outsidePercent > THiop
        inoutAreaCandidate = xor(inoutAreaCandidate, objSeparateLung);
    end
end

getInoutArea = logical(imfill(inoutAreaCandidate, 'hole'));
inoutArea = getInoutArea;