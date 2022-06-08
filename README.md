![img_volumes](https://user-images.githubusercontent.com/73229139/172063936-081317fe-4281-4f33-8803-c4ba3ca6181e.png)

Segmenting by hand an MRI composed of 850 000 voxels takes 90 minutes for a trained expert. The best model presented in this repositery can reconstruct the implant in less than 320 seconds. Its average Dice accuracy is 0.915 on the test set, which translates to an average error of 5.7% for the predicted volume. 

### General Information

### Team
The project is a master semester project conducted by [Julien Adda](https://www.linkedin.com/in/julien-adda-071180195/), a data science student at EPFL. 

The project was supervised by:
- Professor [Dimitri Van de Ville](https://people.epfl.ch/dimitri.vandeville)
- Professor [Cristina Ramona Cudalbu](https://people.epfl.ch/cristina.cudalbu) 
- [Maria Guilia Preti](https://people.epfl.ch/maria.preti/?lang=en)
- [Gninenko Nicolas](https://people.epfl.ch/nicolas.gninenko)

The project was in collaboration with:
- the startup [Volumina SA](https://www.linkedin.com/company/voluminamedical/?originalSubdomain=ch)
- [Center for Biomedical Imaging](https://cibm.ch) (CIBM) at EPFL 
- [Medical Image Processing Lab](https://miplab.epfl.ch) (MIPLab) in Geneva

Contact: julien.adda@epfl.ch

* * *
## Goal of the project

60% of breast cancer patients do not reconstruct their breasts after tumor excision because available procedures are either too invasive or do not bring stable results. The startup Volumina is developing a safe and stable scaffold, Adipearl, that would enable the natural repair of 3D soft tissues in one injection. In order to develop this product, it is crucial to determine the evolution of the injected volume with respect to time. Fortunately, Adipearl is easily recognizable in MRI acquisition, as it is brighter than the other elements of the body. This non invasive technique allows biologist to retrieve the relation between time and the shape of the implant by acquiring several MRI in time. However, labeling by hand the MRI images is time-consuming, which justifies the need to develop an automatic and robust model capable of segmenting the MRI images.

 <img src="https://user-images.githubusercontent.com/73229139/172693699-8a9e7b57-99d4-4148-a4aa-c5c660df9802.png" width="580"/><img src="https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif" width="250" height="270"/>

## Data 
The dataset provided by Volumina SA is composed of 57 MRI T2 weighted images. They were acquired on a 14.1T MRI system at CIBM (located on EPFL’s campus in Lausanne, Switzerland). The data are MRIs of the lower back of 15 mice, where Volumina’s gel, Adipearl, was injected. An MRI acquisition was made one day, three weeks, three months, and six months after the injections for each mouse. Each MRI, is a .nii file of dimensions (46, 192, 96), with a volume of 0.0142 mm3 for each voxel.

## 4 different models
Each model was trained and tested on the same data. The Dice coefficient metric was used to quantify the accuracy of prediction for each model.
- MTOMO : multithresh Otsu method with additional morphological operations (available in the jupyter notebook semi-automatic-methods.ipynb)
- 2d U-net coded from scratch (can be found in the folder 2d U-net from scratch)
- 2d nnU-net 
- 3d nnU-net

nnU-net github: https://github.com/MIC-DKFZ/nnUNet. nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of Helmholtz Imaging.

The ground truth was annotated by hand, slice per slice, by two experts biologists of the Volumina's company.

## Results
### Dice
The dice coefficient on the test set are shown in the following table:

<img width="1333" alt="Capture d’écran 2022-06-08 à 21 28 29" src="https://user-images.githubusercontent.com/73229139/172700696-f52122cf-5611-47b0-bab0-d5183c7e1954.png">

When plotting the dice coefficient per slice (first dimension of the 3D MRI), a U-inverted shape can be seen for each model: 

![dice_per_slice](https://user-images.githubusercontent.com/73229139/172698229-98de4491-f257-4e15-b31e-9258bf2e264d.png)

This indicates that the models are struggling to predict accurately at the extremities (slice 13 to 19 and slice 30 to 35) but achieve a high score in the middle (slice 19 to 30). As the implant is centered in the middle, the drop in accuracy at the extremities is because the implant is ending. The surface to detect is getting smaller, which implies a diminishing contrast between the rare implant pixels and the numerous background pixels and more atypical shapes that the model is not used to detect.

The 3d nnU-net seems to be better for detecting the implant’s extremities. By working in 3d, the model can rely on the previous slices to help locate the implant at the extremities. On the other hand, as the 2d model is treating each slice independently, it cannot anticipate a decrease in volume. The following figure illustrates this particular strength of the 3d model. We can see its capacity to predict the end of the implant accurately compared to the 2d model :


Ground Truth             |  2d nnU-net |  3d nnU-net
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://media.giphy.com/media/f6wc41ySjPSCASIiCB/giphy-downsized-large.gif" width="250" height="270"/>  | <img src="https://media.giphy.com/media/IYmdQbZnsRH4GUX0Sa/giphy-downsized.gif" width="250" height="270"/> | <img src="https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif" width="250" height="270"/>


### Volume

The predicted volume is calculated by counting the voxels labeled 1 in the prediction .nii file. It is then multiplied by the volume of a voxel. 

<img width="939" alt="Capture d’écran 2022-06-08 à 21 32 17" src="https://user-images.githubusercontent.com/73229139/172701558-6bc0f06a-6d91-4f87-a920-67abec2327a7.png">

Ideally, the average would be centered around 0. If it is negative, our model is under-sampling its prediction. In other words, it predicts a smaller volume than the actual volume. It is the case for the two nnU-net (-7.65% and -3.85%). Regarding the absolute average, the closest to 0 the better. We can see that the 3d nnU-net is the best model. 

* * *

## Visualization of results

https://giphy.com/gifs/LFpslzxtwqt1yPJgJu 

![grouped_img](https://user-images.githubusercontent.com/73229139/172698154-2b582937-0717-4f1c-bada-d734ecbcc16b.png)




