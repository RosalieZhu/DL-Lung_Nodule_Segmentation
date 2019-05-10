
function levelSetPhi = Chanvese_InitPhi(inputImg)

%%% This code is made by BAMI LAB %%%
% Level Set Function Initialize

width = size(inputImg, 1);
height = size(inputImg, 2);
levelSetPhi = zeros(width, height);

for j=1:height
    for i=1:width
        levelSetPhi(i, j) = sin(i.*pi./5.0).*sin(j.*pi./5.0);
    end
end

% figure; imshow(levelSetPhi);