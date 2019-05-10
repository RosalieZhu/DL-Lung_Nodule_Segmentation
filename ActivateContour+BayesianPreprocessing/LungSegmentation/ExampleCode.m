
%%% This code is written by Heewon Chung, a member of BAMI LAB %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load Lung Data and Setting Parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('lungData.mat');   % Sample Lung Data
THec = 38;   % Threshold about Ellipse & Circle percent
THiop = 50;   % Threshold about in & out percent

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Chanvese Code 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
getLung = FindGlobalLung(lungData);   % Input the Dicom lung image
getContour = FindContour(getLung);   % Get Contour from get Lung
figure; imshow(getLung);
hold on; plot(getContour{1}(:, 1), getContour{1}(:, 2), 'r', 'LineWidth', 2);   % Left Lung
hold on; plot(getContour{2}(:, 1), getContour{2}(:, 2), 'r', 'LineWidth', 2);   % Right Lung
close all;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Final Lung 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addFinalLung = logical(imadd(inoutArea, getLung));
fillFinalLung = imfill(addFinalLung, 'hole');
finalLung = fillFinalLung;
finalContour = FindContour(finalLung);

clearvars -except preData getLung getContour lungData finalLung finalContour

figure;
subplot(2, 2, 1); imshow(preData);  title('n-1 Final Lung');
subplot(2, 2, 2); imshow(getLung);  title('n Chanvese Lung');
subplot(2, 2, 3); imshow(lungData);  title('n Chanvese Lung');
hold on; plot(getContour{1}(:, 1), getContour{1}(:, 2), 'b', 'LineWidth', 2);
hold on; plot(getContour{2}(:, 1), getContour{2}(:, 2), 'b', 'LineWidth', 2);
subplot(2, 2, 4); imshow(lungData); title('n Final Lung');
hold on; plot(finalContour{1}(:, 1), finalContour{1}(:, 2), 'r', 'LineWidth', 2);
hold on; plot(finalContour{2}(:, 1), finalContour{2}(:, 2), 'r', 'LineWidth', 2);
