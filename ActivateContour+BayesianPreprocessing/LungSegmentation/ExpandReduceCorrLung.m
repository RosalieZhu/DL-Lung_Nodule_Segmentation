
function [erCont, erLung] = ExpandReduceCorrLung(pLung, pCont, nCont)
%%% This code is written by BAMI LAB %%%
%%% Expand and Reduce using prior lung shape 
%%% The prior lung shape is n-1 lung's final result
erLung = false(size(pLung));
erCont = [];

if isempty(nCont) || isempty(pCont)
    return;
end

nData = false(size(pLung));
for nn = 1:length(nCont)
    nv = nCont{nn};
    
    for nnn = 1:size(nv, 1)
        nData(nv(nnn, 2), nv(nnn, 1)) = 1;
    end
end
clear nn nv nnn

rcValue = [5; 4; 3; 2; 1; 1; 2; 3; 4; 5];
reconData = cell(10, 3);
for rr = 1:10
    if rr <= 5
        reduceImg = imerode(pLung, strel('disk', rcValue(rr)));
        maxLung = SeparateMaxLungImages(reduceImg);
        rc = FindContour(maxLung);
        rcCont = rc{1}; clear rc
        rcLine = false(size(pLung));
        for rl = 1:size(rcCont, 1)
            rcLine(rcCont(rl, 2), rcCont(rl, 1)) = 1;
        end
        
        reconData{rr, 1} = reduceImg;
        reconData{rr, 2} = rcCont;
        reconData{rr, 3} = rcLine;
    else
        expandImg = imdilate(pLung, strel('disk', rcValue(rr)));
        maxLung = SeparateMaxLungImages(expandImg);
        rc = FindContour(maxLung);
        rcCont = rc{1}; clear rc
        rcLine = false(size(pLung));
        for rl = 1:size(rcCont, 1)
            rcLine(rcCont(rl, 2), rcCont(rl, 1)) = 1;
        end
        
        reconData{rr, 1} = expandImg;
        reconData{rr, 2} = rcCont;
        reconData{rr, 3} = rcLine;
    end
    clear reduceImg expandImg rcCont maxLung
end
clear rr rc rl rcValue

% contour correlation
rcCount = [];
for rr = 1:10
    rc = reconData{rr, 3};
    
    sameCont = 0;
    for rl = 1:size(pLung, 1)
        for cl = 1:size(pLung, 2)
            if nData(rl, cl) == 1 && rc(rl, cl) == 1
                sameCont = sameCont + 1;
            else
                continue;
            end
        end
    end
    rcCount = [rcCount; sameCont];
    clear rl cl sameCont rc
end
clear rr 

rcIdx = find(rcCount == max(rcCount));
returnLung = reconData{rcIdx, 1};
erCont = FindContour(returnLung);
erLung = returnLung;
