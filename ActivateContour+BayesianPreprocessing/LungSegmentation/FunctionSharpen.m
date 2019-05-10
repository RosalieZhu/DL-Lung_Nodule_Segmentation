
function [sharpenArea] = FunctionSharpen(dicom, angleArea, globalCont)
%%% This code is written by BAMI LAB %%%

pixelData = GetLungAreaData(dicom, angleArea);
objSharp = imsharpen(pixelData, 'Radius', 2, 'Amount', 1);
objSharpAgain = imsharpen(objSharp, 'Radius', 2, 'Amount', 1);
bwObj = im2bw(objSharpAgain);
sharpenRmv = RemoveContourArea(bwObj);

rmvObj = RemoveContourArea(sharpenRmv);
objCont = [];
objImage = false(size(angleArea));
for oo = 1:length(globalCont)
    img = GetImageFromContour(globalCont{oo}, size(angleArea));
    cvimg = bwconvhull(img);
    ovimg = false(size(angleArea));
    ofidx = find(cvimg == 1 & rmvObj == 1);
    ovimg(ofidx) = 1;
    objImage = logical(imadd(objImage, ovimg));
end
objImage = bwareaopen(objImage, 30);
objImage = RemoveContourArea(objImage);

sharpenArea = logical(imfill(objImage, 'holes'));
sharpenCont = FindContour(sharpenArea);