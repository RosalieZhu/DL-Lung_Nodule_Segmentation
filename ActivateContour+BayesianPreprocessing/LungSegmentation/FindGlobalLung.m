
function globalLung = FindGlobalLung(dicom)

%%% This code is made by BAMI LAB %%%
seg = Chanvese_GlobalCode(dicom);

seg2 = seg <= 0;

if seg2(1) == 1
    seg2 = not(seg2);
end

bg = imfill(seg2, 'hole');
lung = xor(bg, seg2);
lung = imfill(lung, 'hole');

allLung = LungSeparation(lung);
globalLung = allLung;
