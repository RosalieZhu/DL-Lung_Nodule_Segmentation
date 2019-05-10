
function lungData = GetLungAreaData(nDicom, lung)
%%% This code is written by BAMI LAB %%%

width = size(nDicom, 1);
height = size(nDicom, 2);
lungData = im2uint8(zeros(width, height));

for ww = 1:width
    for hh=1:height
        if lung(ww, hh) == 1
            lungData(ww, hh) = nDicom(ww, hh);
        else
            continue;
        end
    end
end