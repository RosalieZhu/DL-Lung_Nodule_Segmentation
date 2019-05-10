%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%Series ID and Study ID Mapping%%%%%%%%%%%%%%%%%%%%%%%%

%This file is to get a csv file that maps the Study ID in LIDC-IDRI dataset
%to Series ID in LUNA dataset.

%Author: Nanyan "Rosalie" Zhu and Chen "Raphael" Liu, Columbia University

% Set these paths correctly...
% Input XML file path
%--->change this two directories!!!
LIDC_path   = '/Users/rosaliezhu/Documents/second_semester/projects/Jia_Guo/dataset/LIDC-XML-only'; 
% Where you want to save this result
output_path = '//Users/rosaliezhu/Documents/second_semester/projects/Jia_Guo/dataset/output_ID';  % REPLACE WITH OUTPUT PATH

% Used if no images are found (i.e. you have only downloaded the XML files)
default_pixel_spacing = 0.787109;

% Turns off image missing warnings if using the XML only download
ignore_missing_file_warnings = false;

% Ignore matching warnings when matching GT to images (i.e. if images do 
% not exist, GTs will not be in anatomical order and not all images may be 
% written, this is a 'feature' of the dataset and not an error in the code)
ignore_matching_warnings = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




output_path = correct_path(output_path);
LIDC_path   = correct_path(LIDC_path);

if contains(LIDC_path, ' ')
    error('The LIDC path cannot contain spaces.');
end
if contains(output_path, ' ')
    error('The output path cannot contain spaces.');
end

xml_files = find_files(LIDC_path, '.xml', 'max');
new_xml_paths = cell(1, numel(xml_files)); % Keep a record of the paths that have been processed 
                                           % so that they can be cleaned up if something goes wrong

seriesID_all = [];
studyID_all = [];
for i = 1:numel(xml_files)
    [xml_path, filename] = fileparts(xml_files{i});
    filename = [filename '.xml'];
    xml_path = correct_path(xml_path);
    
    % Extract the individual annotations from the xml files
    [studyID, seriesID] = LIDC_split_annotations_series(xml_path, filename);
    
    seriesID_all = [seriesID_all; seriesID];
    studyID_all = [studyID_all; studyID];
    disp(i)
end

uid_table = table(studyID_all, seriesID_all);
writetable(uid_table,[output_path 'test1.csv']);
