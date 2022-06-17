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


from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
from loss.loss import *
from model.unet import *

import numpy as np
import time
from torch.autograd import Variable
import torch
import os

def training_model(train_loader,loss_function,optimizer,model,num_epochs,model_name,scheduler=None,val_loader = None):
    """Simple training loop for a model, on a training set, with respect to a loss function and optimizer. The function can take a scheduler
    for the learning rate. 

    Args:
        train_loader (DataLoader): Data on which the model will be trained on
        loss_function (LossFunction): Loss function
        optimizer (Optimizer): Optimizer
        model (Model): Model that will be trained
        num_epochs (Int): Number of iterations for training
        scheduler (Scheduler, optional): Schedule the learning rate, see pytorch doc. Defaults to None.

    Returns:
        list: History of train loss
        list: History of val loss
        list: History of train Dice
        list: History of val Dice
    """

    if scheduler == None: # Constant scheduler that does not affect the lr
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1, last_epoch=-1, verbose=False) 
        
    history_train_loss = []
    history_val_loss = []
    history_train_dice = []
    history_val_dice = []
    period = 5

    print("Start training")
    
    for epoch in range(num_epochs+1):
    
        model.train()
        running_train_loss = 0.0
        running_train_dice = []
        t0 = time.time()

        for i, (images,labels,_) in enumerate(train_loader):
               
            #if torch.cuda.is_available():
            #    images=Variable(images.cuda())
            #    labels=Variable(labels.cuda())

            optimizer.zero_grad()
    
            outputs = model(images)
            loss = loss_function(torch.squeeze(outputs), torch.squeeze(labels))
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            if epoch % period == 0:
                for j in range(images.shape[0]):
                    dice_j = dice_coefficient(np.around(sigmoid(outputs[j].detach().cpu().numpy())),labels[j].detach().cpu().numpy())
                    running_train_dice.append(dice_j)
        scheduler.step()


        running_val_loss = 0.0
        running_val_dice = []
        model.eval()

        if (epoch % 100)==0 :
            torch.save(model.state_dict(), 'model/'+ model_name + "_" + str(epoch)+ "epochs" + '.pt')

        if (epoch % period)==0:
            if (val_loader != None):
                for i, (images,labels,_) in enumerate(val_loader):
                    #if torch.cuda.is_available():
                    #                images=Variable(images.cuda())
                    #                labels=Variable(labels.cuda())
                    outputs = model(images)
                    loss = loss_function(torch.squeeze(outputs), torch.squeeze(labels))

                    running_val_loss += loss.item()
                    for j in range(images.shape[0]):
                        dice_j = dice_coefficient(np.around(sigmoid(outputs[j].detach().cpu().numpy())),labels[j].detach().cpu().numpy())
                        running_val_dice.append(dice_j)
                history_val_loss.append(running_val_loss/(len(val_loader)))
                history_val_dice.append(np.mean(running_val_dice))
                print('Epoch n.',epoch, 'Val Loss',np.around(history_val_loss[-1],2), 'Val Dice',np.mean(running_val_dice))
            history_train_loss.append(running_train_loss/(len(train_loader)))
            history_train_dice.append(np.mean(running_train_dice))
            print('Epoch n.',epoch, 'Train Loss',np.around(history_train_loss[-1],2),'Train Dice',np.mean(running_train_dice),'Time Remaining',np.around((num_epochs-epoch)*(time.time()-t0)/60,1),'min')
            print('----------------------------------------------')
    return history_train_loss, history_val_loss, history_train_dice, history_val_dice


