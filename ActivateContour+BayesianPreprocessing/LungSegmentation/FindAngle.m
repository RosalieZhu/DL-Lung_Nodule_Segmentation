
function anglePoint = FindAngle(getLung, getCont)
%%% This code is made by BAMI LAB %%%
% Find Concave point from getLung
%   input  :
%       getLung = binary image
%
%   output :
%       getContour = cell array of lung's contour

if ~isempty(getCont)
    angleValue = [];
    obj = regionprops(getLung);
    calPoint = 5;
    
    for cc = 1:length(getCont)
        nowCont = getCont{cc};
        objCentX = obj(cc).Centroid(1);
        objCentY = obj(cc).Centroid(2);
        
        angleCont = [];
        for gc = 1+calPoint:size(nowCont, 1)-calPoint
            x1 = nowCont(gc-calPoint, 1);
            y1 = nowCont(gc-calPoint, 2);
            x2 = nowCont(gc+calPoint, 1);
            y2 = nowCont(gc+calPoint, 2);
            cx = nowCont(gc, 1);
            cy = nowCont(gc, 2);
            
            newCx = 0;
            newCy = 0;
            newX1 = x1 - cx;
            newY1 = y1 - cy;
            newX2 = x2 - cx;
            newY2 = y2 - cy;
            
            if ((x1 == cx) && (x2 == cx)) || ((y1 == cy) && (y2 == cy))
                angleCont = [angleCont; [cx, cy, 180, 0]];
                continue;
            end
            
            % theta = alpha - beta
            alphaAngle = acosd(newX1/sqrt((newX1.^2) + (newY1.^2)));
            betaAngle = acosd(newX2/sqrt((newX2.^2) + (newY2.^2)));
            if newY1 > 0
                alphaAngle = 360 - alphaAngle;
            end
            if newY2 > 0
                betaAngle = 360 - betaAngle;
            end
            
            theta = abs(alphaAngle - betaAngle);
            if theta < 0.1
                angleCont = [angleCont; [cx, cy, 180, 0]];
                continue;
            end
            thetaK = 0;
            if abs(alphaAngle) > abs(betaAngle)
                if abs(theta) > 360 - abs(theta)
                    thetaK = alphaAngle + ((360 - abs(theta))/2);
                    theta = 360 - abs(theta);
                else
                    thetaK = betaAngle + (theta/2);
                end
            else
                if abs(theta) > 360 - abs(theta)
                    thetaK = betaAngle + ((360 - abs(theta))/2);
                    theta = 360 - abs(theta);
                else
                    thetaK = alphaAngle + (theta/2);
                end
            end
            
            slopeK = -tand(thetaK);
            abSlope = (newY2 - newY1)/(newX2 - newX1);
            abValue = newY2 - (abSlope*newX2);
            kX = abValue/(slopeK - abSlope);
            kY = slopeK*(abValue/(slopeK - abSlope));
            
            if theta == 180
                kX = -newX1;
                kY = slopeK*kX;
            elseif isinf(slopeK)  && (abs(thetaK)==90 || abs(thetaK)==270)
                kX = newCx;
                kY = newY2;
            elseif newY1 == newY2
                kX = newCx;
                kY = newY1;
            elseif newX1 == newX2
                kX = newX1;
                kY = newCy;
            end
            
            primeKx = -kX;
            primeKy = -kY;
            newKx = primeKx + cx;
            newKy = primeKy + cy;
            finalAngle = theta;
            inoutKprime = getLung(floor(newKy), floor(newKx));
            angleCont = [angleCont; [cx, cy, finalAngle, inoutKprime]];
            clear x1 x2 y1 y2 cx cy newCx newCy newX1 newY1 newX2 newY2 alphaAngle betaAngle theta thetaK slopeK kX kY primeKx primeKy newKx newKy primeAngle objCenterAngle calKx calKy calObjCentX calObjCentY finalAngle abSlope abValue
        end
        angleValue{cc, 1} = angleCont;
        clear gc angleCont
    end
    clear cc objCentX objCentY calNum obj nowCont
    
    for aa = 1:length(angleValue)
        cidx = find(angleValue{aa}(:, 3) < 140 & angleValue{aa}(:, 4) == 1);
        cutValue = angleValue{aa}(cidx, 1:2);
        anglePoint{aa, 1} = cutValue;
        clear cidx cutValue
    end
else
    anglePoint = [];
end