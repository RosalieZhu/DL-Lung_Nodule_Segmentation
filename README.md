# DL-Lung_Nodule_Segmentation
### This is a repository for deep learning course project in Columbia University

Team R&R: Nanyan "Rosalie" Zhu & Chen "Raphael" Liu 

Please read our [Paper](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/End-to-End%20Lung%20Nodule%20Segmentation%20and%20Visualization%20in%20Computed%20Tomography%20using%20Attention%20U-Net.pdf) first

## Preprocessing
1. Change your LUNA16 data directory in "prepeocessing" folder. [Processing_LungMask_ActiveContourBayesian_Part1_RunAll_Linux.m](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/ActivateContour%2BBayesianPreprocessing/Processing_LungMask_ActiveContourBayesian_Part1_RunAll_Linux.m) generates preprocessed and segmented lung dicom scans, and [Processing_LungMask_ActiveContourBayesian_Part2_RunAll.m](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/ActivateContour%2BBayesianPreprocessing/Processing_LungMask_ActiveContourBayesian_Part2_RunAll.m) generates lung nodule dicom scans.

## Visualize the data we have
2. [Radiologists_labels.ipynb](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/visulization/Radiologists_labels.ipynb) in visualization folder analyzed our oringinal scans and oringinal labels of radiologists. The ground truth labels were got from LIDC dataset converted by [LIDCtoolbox](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/tree/master/Supplementary/LIDCToolbox) in supplementary folder.

## Generate patches
3. [scans_to_images.ipynb](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/Supplementary/scans_to_images.ipynb) in supplementary folder generates patches for patch based U-Net family training.
### Data Augmentation Demo
![](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/Fig/Data_augmentation.png)

## Network
4. Run [patch-based_Unet_main.ipynb](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/Network/patch-based_Unet_main.ipynb) in Network folder to get models trained, and use [result_checking](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/visulization/result_checking) (for different models) and [result_checking-inverserate](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/visulization/result_checking-inverserate) (for different inverse ratio $\lambda$ in our report) to check [test statistics](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/visulization/result_plot.ipynb) of those models.

## Activation Map

![](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/Fig/layer1_filters.png)
![](https://github.com/RosalieZhu/DL-Lung_Nodule_Segmentation/blob/master/Fig/Activation_maps.png)

## Result
![](lung-segmented.gif)          ![](Seg_result.gif)


