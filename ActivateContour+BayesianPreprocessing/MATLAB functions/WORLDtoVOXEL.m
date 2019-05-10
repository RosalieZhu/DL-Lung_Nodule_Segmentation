% WORLDtoVOXEL Converting world coordinates to voxel coordinates for the 3D CT lung scans.
%
% world_coordinates: the coordinates of nodule annotations in mm.
% voxel_spacing: the spacing between adjacent voxels [spacing_x, spacing_y, spacing_z] with units in mm.
% it is given as the "spacing" parameter in the metadata of the *.mhd file.
% scan_origin: the origin of the 3D CT scan. It is given as the "origin" parameter in the metadata of the *.mhd file.
%
% April 19, 2019.
% Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
% Columbia University

function voxel_coordinates = WORLDtoVOXEL(world_coordinates, voxel_spacing, scan_origin)
    voxel_coordinates = (world_coordinates - scan_origin) ./ voxel_spacing;
end