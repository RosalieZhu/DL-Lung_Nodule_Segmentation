
function angleArea = RemainAngleArea(diffArea, anglePoint)
%%% This code is written by BAMI LAB %%%
%%% Find angle Points for difference Area 

% Object Detection
obj = regionprops(diffArea);
angleArea = false(size(diffArea));

if ~isempty(obj)
    width = size(diffArea, 1);
    height = size(diffArea, 2);
    angleImg = GetAngleImage(anglePoint, width, height);
    expandImg = imdilate(diffArea, strel('disk', 2));
    dfCont = FindContour(expandImg);
    
    if isempty(find(angleImg == 1)) 
        return;
    end
    
    rmvImg = [];
    for df = 1:length(dfCont)
        dfData = dfCont{df};
        dfArea = GetImageFromContour(dfData, size(diffArea));
        aidx = find(angleImg == 1);
        didx = find(dfArea == 1);
        
        fcount = 0;
        for dd = 1:size(aidx, 1)
            sidx = find(didx == aidx(dd));
            if ~isempty(sidx)
                fcount = fcount + 1;
            end
        end
        
        if fcount > 0
            rmvImg = [rmvImg; {dfArea}];
        end
    end
    
    addImg = false(size(diffArea));
    for rr = 1:length(rmvImg)
        addImg = logical(imadd(addImg, rmvImg{rr}));
    end
    
    selObj = false(size(diffArea));
    selIdx = find(addImg == 1 & diffArea == 1);
    selObj(selIdx) = 1;    
    angleArea = selObj;
else
    return;
end
