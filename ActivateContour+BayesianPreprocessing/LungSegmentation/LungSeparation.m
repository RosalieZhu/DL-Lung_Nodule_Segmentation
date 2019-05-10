
function [rmLung] = LungSeparation(lung)

sobj = regionprops(logical(lung));
rmLung = false(size(lung));
if isempty(sobj)
    return;
end

[sortArea, sidx]  = sort([sobj.Area], 'descend');
sMaxArea = sortArea(1);
sPctArea = sMaxArea * 0.05;
rmArea = [];
for rr=1:size(sobj, 1)
    if sortArea(rr) > sPctArea
        rmArea = [rmArea; sortArea(rr)];
    end
end
clear sobj sortArea sidx sMaxArea sPctArea

rmLung =  bwareaopen(lung, rmArea(end));
rmLung = logical(rmLung);
