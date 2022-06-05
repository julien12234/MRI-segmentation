![img_volumes](https://user-images.githubusercontent.com/73229139/172063936-081317fe-4281-4f33-8803-c4ba3ca6181e.png)


The repository contains the code and a peek to some of the results. 

Several techniques were tested, starting by simple algorithm such as Multi-Otsu Thresholding and Canny edge detection. A deeper analysis was made by training Conventional Neural Network model based on U-net \cite{chhor2017satellite}. A simple 2d architecture with a BCE loss function  was compared to a 2d and 3d nnU-net proposed by the Applied Computer Vision Lab (ACVL) of Helmholtz Imaging. The models were trained to do binary pixel-wise segmentation on MRI .nii files. The labels to predict are binary masks, where the pixels representing the implant are ones, and zero otherwise. The output of the models are a binary .nii file, which allows us to compute the volume of the implant. 

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

### Environment
The project has been developed and test with `python3.6`.

The required library are `numpy, Pytorch, sklearn, openCV`

The library for visualization is `matplotlib`.

* * *
## Goal of the project

60% of breast cancer patients do not reconstruct their breasts after tumor excision because available procedures are either too invasive or do not bring stable results. The startup Volumina is developing a safe and stable scaffold, Adipearl, that would enable the natural repair of 3D soft tissues in one injection. In order to develop this product, it is crucial to determine the evolution of the injected volume with respect to time. Fortunately, Adipearl is easily recognizable in MRI acquisition, as it is brighter than the other elements of the body. This non invasive technique allows biologist to retrieve the relation between time and the shape of the implant by acquiring several MRI in time. However, labelling by hand the MRI images is time consuming, which justify the need to develop an automatic and robust model capable of segmenting the MRI images. 

The goal of the project is to devellop a fully automated techniques capable of segmenting an MRI images, and reconstruct the shape of the implant. 

<img width="500" height="300" src="https://user-images.githubusercontent.com/73229139/172062769-683f06c4-37db-40e6-9eca-404f4544d7f7.png"> <img src="https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif" width="275" height="250"/>

## Data 

The dataset provided by Volumina SA is composed of 57 MRI T2 weighted images. They were acquired on a 14.1T MRI system at CIBM (located on EPFL's campus in Lausanne, Switzerland). The data are MRIs of the lower back of 15 mice, where Volumina's gel, Adipearl, was injected. For each mice, a MRI acquisition was made 1 day, 3 weeks 3 months and 6 months after the injections. Each MRI, is a .nii file of dimensions (46, 192, 96), with a volume of 0.0142 $mm^3$ for each voxel. The ground truth are also .nii files of the same dimensions. They contain  values of 1 or 0, indicating respectively the presence of a voxel of Adipearl or the background.  

The ground truth was annotated by hand, slice per slice, by two experts biologists of the Volumina's company.

### MTOMO

![semi_auto_img](https://user-images.githubusercontent.com/73229139/172062095-7b08c504-1cf5-44c0-9655-bcecbbb8a967.png)

The project target is to segment in aerial images of Switzerland(Geneva) the area available for the installation of rooftop photovoltaics (PV) panels, namely the area we have on roofs after excluding chimneys, windows, existing PV installations and other so-called ‘superstructures’. The task is a pixel-wise binary-semantic segmentation problem. And we are interested in the class where pixels can be classified as ‘suitable area’ for PV installations.

![Screenshot from 2020-12-16 13-11-43](https://user-images.githubusercontent.com/32882147/102347151-47643980-3fa0-11eb-83c7-354c90462914.png)

### Data
- The input aerial images are RGB aerial images in PNG form and  each  image  has  size 250×250×3 with pixelsize 0.25×0.25 m^2. 
- We used the provided labelling tool to manually label all the data The labelled images are a binary mask with 1 for pixel in PV area, and 0 otherwise.
- The original input images are transformed with saturation and classic normalization before training. 
- A real-time data argumentation is applied only on the training set by randomly flipping images horizontally or vertically or rotating in ninety degrees.
- The  output  of  our  model  is again a binary image, where the pixel is one, if its probability of being in the PV area is bigger than a fixed threshold.
- Train/Validation/Test Ratio : 80/10/10 \%


Ground Truth             |  2d nnU-net |  3d nnU-net
:-------------------------:|:-------------------------:|:-------------------------:
![Alt Text](https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif)  |  ![Alt Text](https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif) |  ![Alt Text](https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif)


* * *
## Project structure
```bash
├── labelling_tool
│   ├── crop.py
│   ├── data-verification.ipynb
│   ├── label_images_from_txt.py
│   ├── label_images.py
│   ├── move.py
│   ├── README.md
│   └── scan_images.py
├── loss
│   ├── 1.png
│   ├── 2.png
│   ├── loss.ipynb
│   ├── loss.py
│   └── __pycache__
│       └── loss.cpython-38.pyc
├── main.ipynb
├── model
│   ├── model.ipynb
│   ├── __pycache__
│   │   └── unet.cpython-38.pyc
│   └── unet.py
├── plots
│   ├── all
│   │   ├── batch5loss4
│   │   │   ├── history_train_ioubatch5loss4_1000.npy
│   │   │   ...
│   │   │   └── loss400_batch5loss4.png
│   │   └── batch5loss9
│   │       ├── history_train_ioubatch5loss9_1000.npy
│   │       ... 
│   │       └── loss1000_batch5loss9.png
│   ├── other
│   │   ├── b5w4_iou.pdf
│   │   ├── b5w4_loss.pdf
│   │   ├── history_train_iou09122020.npy
│   │   ...
│   │   ├── loss_batch5loss4.pdf
│   │   └── loss_batch5loss5.pdf
│   ├── plots.py
│   └── residencial
│       ├── history_train_ioubatch5loss6_1000.npy
│       ...
│       ├── loss1000_batch5loss6.png
│       └── loss_batch5loss6.png
├── process_data
│   ├── data_loader.py
│   ├── import_test.py
│   ├── normalize.py
│   └── __pycache__
│       ├── data_loader.cpython-38.pyc
│       ├── data_noara_loader.cpython-38.pyc
│       └── data_nopv_loader.cpython-38.pyc
├── README.md
├── reference
│   └── Literature
│       ├── Adam a method \for stochastic optimization.pdf
│       ├── Deep learning \in the built environment automatic detection of rooftop solar panels 
│           using Convolutional Neural Networks.pdf
│       ├── Dropout vs. batch normalization an empirical study.pdf
│       ├── Satellite Image Segmentation \for Building Detection using U-Net.pdf
│       ├── Semantic Segmentation of Satellite Images using Deep Learning.pdf
│       └── U-Net, Convolutional Networks \for Biomedical Image Segmentation.pdf
├── run.py
└── train
    ├── pred_residencial_3.png
    └── train.py
```




https://giphy.com/gifs/LFpslzxtwqt1yPJgJu 
