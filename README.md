# Development of a segmentation tool to measure subcutaneous implant volumes

The repository contains the code and a peek to the results. 

* * *
### General Information

### Team
The project is conducted by Julien Adda, supervised by:
- Professor Dimitri Van de Ville
- Cristina Ramona Cudalbu
- Maria Guilia Preti 
- Gninenko Nicolas


### Environment
The project has been developed and test with `python3.6`.

The required library are `numpy, Pytorch, sklearn, openCV`

The library for visualization is `matplotlib`.

* * *
## Data

<img width="450" height="400" alt="Capture d’écran 2022-06-05 à 19 12 26" src="https://user-images.githubusercontent.com/73229139/172062144-0b2a5d35-69a9-4348-bc81-73d1ec3b7150.png"> <img src="https://media.giphy.com/media/LFpslzxtwqt1yPJgJu/giphy-downsized.gif" width="400" height="400"/>


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




https://giphy.com/gifs/LFpslzxtwqt1yPJgJu 
