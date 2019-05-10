# DL-Lung_Nodule_Segmentation
### This is a repository for BMEN4000 course project in Columbia University

Team R&R: Nanyan "Rosalie" Zhu (nz2305) & Chen "Raphael" Liu (cl3760)

## Preprocessing
1. Change your LUNA16 data directory in "prepeocessing" folder. "Processing_LungMask_ActiveContourBayesian_Part1_RunAll_Linux.m" generates preprocessed and segmented lung dicom scans, and "Processing_LungMask_ActiveContourBayesian_Part2_RunAll.m" generates lung nodule dicom scans.

## Visualize the data we have
2. "Radiologists_labels.ipynb" in visualization folder analyzed our oringinal scans and oringinal labels of radiologists. The ground truth labels were got from LIDC dataset converted by LIDCtoolbox in supplementary folder.

## Generate patches
3. "scans_to_images.ipynb" generates patches for patch based U-Net family training.

## Network
4. Run "patch-based_Unet_main.ipynb" in Network folder to get models trained, and use "result_checking" (for different models) and "result_checking-inverserate" (for different inverse ratio $\lambda$ in our report) to check test statistics of those models.
