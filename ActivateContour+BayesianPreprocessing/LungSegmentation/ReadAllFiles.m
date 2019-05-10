
function [files] = ReadAllFiles(folderPath, ext)

% This method returns all dcm files to search all folders
% 
%   input : 
%       folderPath = The path of folder that located files to find
%       ext        = The extension of file to find


dirData = dir(folderPath);
dirIdx = [dirData.isdir];

files = [];
fileList = {dirData(~dirIdx).name}';

if ~isempty(fileList)
    files = GetFiles(fileList, ext);
    files = cellfun(@(x) fullfile(folderPath,x),...
                       files, 'UniformOutput', false);
end

subDirs = {dirData(dirIdx).name};
validIdx = ~ismember(subDirs, {'.', '..'});

for iDir = find(validIdx)
    nextDir = fullfile(folderPath, subDirs{iDir});
    files = [files; GetAllFiles(nextDir, ext)];
end

% -----------------------------------------------------------------------

function [files] = GetFiles(files, ext)

count = length(files);
idx = zeros(count,1);

for i=1:count
    [~,~,e] = fileparts(files{i});
    
    if isequal(e, strcat('.', ext))
        idx(i) = 1;
    else
        idx(i) = 0;
    end
end

files = files(idx~=0);