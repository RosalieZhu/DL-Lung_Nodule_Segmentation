# LungSegmentation
Automatic Lung Segmentation with Juxta-Pleural Nodule Identification Using Active Contour Model and Bayesian Approach
      

ABSTRACT :      
- Objective: chest computed tomography (CT) images and their quantitative analyses have become increasingly important for a variety of purposes, including lung parenchyma density analysis, airway analysis, diaphragm mechanics analysis, and nodule detection for cancer screening. Lung segmentation is an important prerequisite step for automatic image analysis. We propose a novel lung segmentation method
to minimize the juxta-pleural nodule issue, a notorious challenge in the applications. 
- Method: we initially used the Chan–Vese (CV) model for active lung contours and adopted a Bayesian approach based on the CV model results, which predicts the lung image based on the segmented lung contour in the previous frame image or neighboring upper frame image. Among the resultant juxta-pleural nodule candidates, false positives were eliminated through concave points detection and circle/ellipse Hough transform. Finally, the lung contour was modified by adding the final nodule candidates to the area of the CV model results.
- Results: to evaluate the proposed method, we collected chest CT digital imaging and communications in medicine images of 84 anonymous subjects, including 42 subjects with juxta-pleural nodules. There were 16 873 images in total. Among the images, 314 included juxta-pleural nodules. Our method exhibited a disc similarity coefficient of 0.9809, modified hausdorff distance of 0.4806, sensitivity of 0.9785, specificity of 0.9981, accuracy of 0.9964, and juxta-pleural nodule detection rate of 96%. It outperformed existing methods, such as the CV model used alone, the normalized CV model, and the snake algorithm. Clinical impact: the high accuracy with the juxta-pleural nodule detection in the lung segmentation can be beneficial for any computer aided diagnosis system that uses lung segmentation as an initial step.

Main Code :
- ExampleCode.m   
- ExampleDetailCode.m

Description (Change parameter in ExampleDetailCode.m)
- Line 6, FolderPath : Dicom file path
- Line 21~22, Width & Center : Window setting related Dicom images
- See the article 'Automatic Lung Segmentation with Juxta-Pleural Nodule Identification Using Active Contour Model and Bayesian Approach'.
      
