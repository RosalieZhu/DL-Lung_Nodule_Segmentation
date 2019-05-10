
%%% This code is written by Heewon Chung, a member of BAMI LAB %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load Lung Data and Setting Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
folderPath = 'D:\Lung Segmentation\DicomLungn\';
dcmfiles = ReadAllFiles(folderPath, 'dcm');
dcmLength = length(dcmfiles);
dcmLength = 2;

THec = 38;   % Threshold about Ellipse & Circle percent
THiop = 50;   % Threshold about in & out percent
lungFinalData = cell(dcmLength, 1);
lungFinalCont = cell(dcmLength, 1);

for sliceIndex = 1:dcmLength
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Chanvese Code
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %=== window setting related Dicom images
    %=== change window setting (related Dicom images)
    width = 1370;
    center = 750;
    low = center - round(width/2);
    high = center + round(width/2);
    
    dicomImg = dicomread(dcmfiles{sliceIndex});
    im = mat2gray(dicomImg, [low high]);
    im = im2uint8(im);
    lungData = im;
    lungData = scan(:,:,50+sliceIndex);
    getLung = FindGlobalLung(lungData);   % Input the Dicom lung image
    getContour = FindContour(getLung);   % Get Contour from get Lung
    %     figure; imshow(getLung);
    %     hold on; plot(getContour{1}(:, 1), getContour{1}(:, 2), 'r', 'LineWidth', 2);   % Left Lung
    %     hold on; plot(getContour{2}(:, 1), getContour{2}(:, 2), 'r', 'LineWidth', 2);   % Right Lung
    %     close all;
    
    if sliceIndex == 1
        lungFinalData{sliceIndex, 1} = getLung;
        lungFinalCont{sliceIndex, 1} = getContour;
        sliceIndex = sliceIndex + 1;
    end
    
    % Get Data
    preData = lungFinalData{sliceIndex-1};
    preContour = lungFinalCont{sliceIndex-1};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Filtering using Chanvese result lung
    %%% juxta-pleural nodule segmentation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    anglePoint = FindAngle(getLung, getContour);   % Find concave point in binary lung image
    [erContour, erLung] = ExpandReduceCorrLung(preData, preContour, getContour);    % get Expand and Reduce Model(erModel) from n-1 final Lung
    diffArea = DifferenceArea(erLung, getLung);   % difference Area of erModel Lung and Current Lung
    angleArea = RemainAngleArea(diffArea, anglePoint);   % Find angle Points for difference Area
    sharpenArea = FunctionSharpen(lungData, angleArea, getContour);   % Sharpness
    ecArea = EllipseAndCircularCorrelation(sharpenArea, THec);   % get Ellipse and Circular Correlation
    inoutArea = FunctionInoutLungArea(ecArea, getLung, THiop);   % get In & Out calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Final Lung
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    addFinalLung = logical(imadd(inoutArea, getLung));
    fillFinalLung = imfill(addFinalLung, 'hole');
    finalLung = fillFinalLung;
    finalContour = FindContour(finalLung);
    lungFinalData{sliceIndex} = finalLung;
    lungFinalCont{sliceIndex} = finalContour;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    figure;
    subplot(2, 2, 1); imshow(preData);  title('n-1 Final Lung');
    subplot(2, 2, 2); imshow(getLung);  title('n Chanvese Lung');
    subplot(2, 2, 3); imshow(lungData);  title('n Chanvese Lung');
    hold on; plot(getContour{1}(:, 1), getContour{1}(:, 2), 'b', 'LineWidth', 2);
    hold on; plot(getContour{2}(:, 1), getContour{2}(:, 2), 'b', 'LineWidth', 2);
    subplot(2, 2, 4); imshow(lungData); title('n Final Lung');
    hold on; plot(finalContour{1}(:, 1), finalContour{1}(:, 2), 'r', 'LineWidth', 2);
    hold on; plot(finalContour{2}(:, 1), finalContour{2}(:, 2), 'r', 'LineWidth', 2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end