
function rmvContourArea = RemoveContourArea(rmvCircle)
%%% This code is written by BAMI LAB %%%

rmvLine = logical(imfill(rmvCircle, 'holes'));

for rr = 1:5
    valueFind = find(rmvLine == 1);
    wPoint = [];
    for ii = 1:length(valueFind)
        ww = size(rmvLine, 1);
        indx = valueFind(ii);
        x = fix(indx/ww)+1;
        y = rem(indx, ww);
        wPoint = [wPoint; [x, y]];
        clear ww indx x y
    end
    clear ii
    
    for ii = 1:size(wPoint, 1)
        px = rmvLine(wPoint(ii, 2)-1:wPoint(ii, 2)+1, wPoint(ii, 1)-1:wPoint(ii, 1)+1);
        fp = find(px == 1);
        fl = length(fp);
        if fl < 5
            rmvLine(wPoint(ii, 2), wPoint(ii, 1)) = 0;
        end
    end
end

getObj = bwareaopen(rmvLine, 50);
rmvContourArea = logical(imfill(getObj, 'hole'));