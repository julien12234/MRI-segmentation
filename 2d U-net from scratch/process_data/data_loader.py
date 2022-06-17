#    Copyright 2022 Julien Adda
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.utils.data as data
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import glob
import numpy as np
import os
from PIL import Image
import torch
import random
import cv2


class DataLoaderSegmentation(data.Dataset):
    """Class for all three data loaders (train, test, val)
    """
    
    def __init__(self, folder_path_img,folder_path_mask=None):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """

        self.img_files = glob.glob(os.path.join(folder_path_img,'*.png'))

        if folder_path_mask == None:
            self.mask_files = 0
        else:
            self.mask_files =glob.glob(os.path.join(folder_path_mask,'*.png'))
        

    def transform(self, image, mask):
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.4066, 0.4768, 0.4383],std=[0.2121, 0.1899, 0.1618]) # For All
        
        mask = TF.to_tensor(mask)
        mask = mask[0]
        mask = mask > 0
        mask = mask.float()
                
        return image, mask

    def __getitem__(self, index,show_og=False):
        """Get specific data corresponding to the index applying randomly dat augmentation
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        
        """
        # GET IMAGE
        """
        image = Image.open(self.img_files[index]).convert('RGB')
        if self.mask_files != 0:
            mask = Image.open(self.mask_files[index])
        else:
            mask_shape = image.size
            mask = Image.new('RGB', mask_shape)        
        
        z = image
        x, y = self.transform(image, mask)

        if show_og:
            return x, y, z, self.img_files[index]
        else:
            return x, y, self.img_files[index]

    def __len__(self):
        return len(self.img_files)

