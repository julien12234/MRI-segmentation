![img_volumes](https://user-images.githubusercontent.com/73229139/172063936-081317fe-4281-4f33-8803-c4ba3ca6181e.png)

Segmenting by hand, an MRI composed of 850 000 voxels takes 90 minutes for a trained expert. The best model presented in this repository can reconstruct the implant in less than 320 seconds. Its average Dice accuracy is 0.915 on the test set, which translates to an average error of 5.7% for the predicted volume. 

### Findings 

The top-performing algorithm is currently in use at the CIBM laboratory in Lausanne. The results of our research were published on August 21, 2023, in the scientific journal Frontiers in Signal Processing, with me as the primary author.

Title: [Automated segmentation and labeling of subcutaneous mouse implants at 14.1T](https://doi.org/10.3389/frsip.2023.1155618)

### Report

`documents/report.pdf`: an 12-pages report of this project

### Team
The project is a master semester project conducted by [Julien Adda](https://www.linkedin.com/in/julien-adda-071180195/), a data science student at EPFL. 

The project was supervised by:
- Professor [Dimitri Van de Ville](https://people.epfl.ch/dimitri.vandeville)
- Professor [Cristina Ramona Cudalbu](https://people.epfl.ch/cristina.cudalbu) 
- Dr. [Gninenko Nicolas](https://people.epfl.ch/nicolas.gninenko)
- Dr. [Maria Guilia Preti](https://people.epfl.ch/maria.preti/?lang=en)


The project was in collaboration with:
- the startup [Volumina SA](https://www.linkedin.com/company/voluminamedical/?originalSubdomain=ch)
- [Center for Biomedical Imaging](https://cibm.ch) (CIBM) at EPFL 
- [Medical Image Processing Lab](https://miplab.epfl.ch) (MIPLab) in Geneva

Contact: julien.adda@gmail.com

* * *
## Goal of the project

60% of breast cancer patients do not reconstruct their breasts after tumor excision because available procedures are either too invasive or do not bring stable results. The startup Volumina is developing a safe and stable scaffold, Adipearl, that would enable the natural repair of 3D soft tissues in one injection. To develop this product, it is crucial to determine the evolution of the injected volume with respect to time. Fortunately, Adipearl is easily recognizable in MRI acquisition, as it is brighter than the other elements of the body. This non-invasive technique allows biologists to retrieve the relation between time and the shape of the implant by acquiring several MRIs in time. However, labeling by hand the MRI images is time-consuming (90 minutes per MRI). 

**The goal of this project is to develop an automatic and robust model capable of segmenting the MRI images.** 



 <img src="https://user-images.githubusercontent.com/73229139/172693699-8a9e7b57-99d4-4148-a4aa-c5c660df9802.png" width="580"/><img src="https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif" width="250" height="270"/>

## Data 
The dataset provided by Volumina SA is composed of 57 MRI T2 weighted images. They were acquired on a 14.1T MRI system at CIBM (located on EPFL’s campus in Lausanne, Switzerland). The data are MRIs of the lower back of 15 mice, where Volumina’s gel, Adipearl, was injected. An MRI acquisition was made one day, three weeks, three months, and six months after the injections for each mouse. Each MRI, is a .nii file of dimensions (46, 192, 96), with a volume of 0.0142 mm3 for each voxel.

## 4 different models
Each model was trained and tested on the same data. The Dice coefficient metric was used to quantify each model's prediction accuracy.
- **MTOMO:** multithresh Otsu method with additional morphological operations (available in the jupyter notebook `semi-automatic-methods.ipynb`)
- **2d U-net:** coded from scratch (can be found in the folder 2d U-net from scratch)
- **2d nnU-net**
- **3d nnU-net**

nnU-net Github: https://github.com/MIC-DKFZ/nnUNet. nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of Helmholtz Imaging.

The ground truth was annotated by hand, slice per slice, by two expert biologists of the Volumina's company.

## Run the best model

### Steps
The pre-trained model 3d nnU-net is available, and 1 MRI and the ground truth: it is mouse 92-779-4. The MRI was acquired one day after the injection of Adipearl. A python file, `CIBM_prediction.py`, can be run to test the model on the data located in the input/ file. 

Here are the steps to test the model:

<ins>Do one time:</ins>
- Make sure to have miniconda (python=3.7): https://docs.conda.io/en/latest/miniconda.html 
- Download the pre-trained model, the data & `CIBM_prediction.py` at this [link.](https://drive.google.com/drive/folders/1L_ou2JzvqUIT4g4j3ZgCgyOEqLLGt7M6?usp=sharing)
- Go in the directory of the downloaded folder and open a terminal 
- Create a python environment: `conda create -n "Name_of_python_environment" python=3.7.13 ipython*`
- Activate the created python environment: `conda activate Name_of_python_environment`
- Install the requirements: `pip install -r requirements.txt`. It will install the following libraries: 
  - SimpleITK
  - colorama==0.4.4
  - matplotlib==3.5.1
  - torch==1.7.1
  - nnunet==1.7.0

<ins>Do every time you want to predict a .nii.gz file:</ins>
- Go in the directory of the downloaded folder and open a terminal 
- Activate the created python environment: `conda activate Name_of_python_environment`
- Respect the rules of the file `CIBM_prediction.py` (located at the beginning of the file and presented in the next section):
  - put your files to be predicted in the `input/` folder
  - empty your `output/` folder
  - if needed, add your label file in the `label/` folder
- Run file `CIBM_prediction.py` from the terminal with the command: `python CIBM_prediction.py` 
- Answer Yes or No to the question: "Check label folder (Yes/No):"

### Rules and structure: `CIBM_prediction.py`

This script is used to segment .nii or .nii.gz files located in the folder `input/`. The predictions will be located in the output folder `output/`, with the prefix `prediction_` in front of the input files.

It will use a 3d trained nnU-net model (https://github.com/MIC-DKFZ/nnUNet).

Please, before running the script, make sure to:
- install all the required libraries in the correct version 
- have only .nii.gz files of dimensions (96, 192, 46) in the folder_input
- empty your folder_output 
- if ground truth are added in the label/ folder: add the prefix "label_" to the names of the .nii.gz file. 

Your directory should be structured in the following way:
```
├──  CIBM_prediction.py        -> file to be runned, to test data located in input/ folder 
├──  requirements.txt          -> requirements to install
├──  liscence     
├──  input/                    !!!Input files need to be .nii.gz of dimenion (96, 192, 46)!!!
│    ├── image1.nii.gz
│    ├── image2.nii.gz
│    ├── image3.nii.gz
├──  output/                   !!!Keep this folder empty before running the script!!!
├──  label/                    Contains the label of the MRI in input -> always add label_ in front of the name
│    ├── label_image1.nii.gz
│    ├── label_image2.nii.gz
├──  nnUNet/                   !!!Do not modify this folder!!!
```

The output folder, after prediction, will be:
```
├──  output/                
│    ├── prediction_image1.nii.gz
│    ├── prediction_image2.nii.gz
│    ├── prediction_image3.nii.gz
```

### Output

If you ask to check the `label/` folder and some files were found, the output in the terminal will look like this example where 5 files were found:

<img width="1239" alt="Capture d’écran 2022-06-07 à 14 21 43" src="https://user-images.githubusercontent.com/73229139/174307167-33aa725d-3bcf-4b8a-a9f9-031a16ce2562.png">

If you did not ask to check the `label/` folder: 

<img width="1239" alt="Capture d’écran 2022-06-17 à 15 42 34" src="https://user-images.githubusercontent.com/73229139/174310177-3e9f7216-6ba3-4ef3-868e-4b4081ccece5.png">

## Results
### Dice
The dice coefficient on the test set are shown in the following table:

![Capture d’écran 2023-02-15 à 12 41 58](https://user-images.githubusercontent.com/73229139/219018680-99f2a065-0f59-4eb1-83d3-53c7292d58e5.png)

When plotting the dice coefficient per slice (first dimension of the 3D MRI), a U-inverted shape can be seen for each model: 

![dice_per_slice_evo](https://user-images.githubusercontent.com/73229139/172791617-e3a0f713-35b4-44ee-8a51-b323ab9afc18.png)


This indicates that the models struggle to predict accurately at the extremities (slice 13 to 19 and slice 30 to 35) but achieve a high score in the middle (slice 19 to 30). As the implant is centered in the middle, the drop in accuracy at the extremities is because the implant is ending. The surface to detect is getting smaller, which implies a diminishing contrast between the rare implant pixels and the numerous background pixels and more atypical shapes that the model is not used to detect.

The 3d nnU-net seems to be better for detecting the implant’s extremities. By working in 3d, the model can rely on the previous slices to help locate the implant at the extremities. On the other hand, as the 2d model is treating each slice independently, it cannot anticipate a decrease in volume. The following figure illustrates this particular strength of the 3d model. We can see its capacity to predict the end of the implant accurately compared to the 2d model :


Ground Truth             |  2d nnU-net |  3d nnU-net
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://media.giphy.com/media/f6wc41ySjPSCASIiCB/giphy-downsized-large.gif" width="250" height="270"/>  | <img src="https://media.giphy.com/media/IYmdQbZnsRH4GUX0Sa/giphy-downsized.gif" width="250" height="270"/> | <img src="https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif" width="250" height="270"/>


### Volume

The predicted volume is calculated by counting the voxels labeled 1 in the prediction .nii file. It is then multiplied by the volume of a voxel. 

<img width="939" alt="Capture d’écran 2022-06-08 à 21 32 17" src="https://user-images.githubusercontent.com/73229139/172701558-6bc0f06a-6d91-4f87-a920-67abec2327a7.png">

Ideally, the average would be centered around 0. If it is negative, our model is under-sampling its prediction. In other words, it predicts a smaller volume than the actual volume. It is the case for the two nnU-net (-7.65% and -3.85%). Regarding the absolute average, the closest to 0, the better. We can see that the 3d nnU-net is the best model. 

* * *

## Visualization of results

Ten examples of prediction on the test set by the different models. The MRI is the first image, and the concerned slice and mouse id are written at the top. The red area is the implant to be segmented. The ground truth, annotated by hand, is the last image. The Dice accuracy is shown at the top of each image. We can see the robustness of the 3d nnU-net compared to the other models.

![grouped_img](https://user-images.githubusercontent.com/73229139/172698154-2b582937-0717-4f1c-bada-d734ecbcc16b.png)


## License
-------

Copyright 2022, Julien Adda.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

