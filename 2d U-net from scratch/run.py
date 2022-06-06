import numpy as np
import matplotlib.pyplot  as plt
import torch
from torch.autograd import Variable
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset
from train.train import *
from tempfile import TemporaryFile

from model.unet import *
from loss.loss import *
from process_data.data_loader import *
from plots.plots import * 


def seed_torch(seed=0):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ ==  '__main__':

    device = "cpu"
    seed_torch() 

    # Path to your data (2D png images of dimenions (192 · 96))
    # ground-truth are binary images (1 or 0), mask of the images 
    # ground-truth and image need to have the same name
    folder_path_train_image = 'data/train/images'
    folder_path_train_masks = 'data/train/ground-truth'
    folder_path_test_image = 'data/test/images'
    folder_path_test_masks = 'data/test/ground-truth'
    folder_path_val_image = 'data/val/images'
    folder_path_val_masks = 'data/val/ground-truth'

    # Load dataset
    train_set = DataLoaderSegmentation(folder_path_train_image, folder_path_train_masks) 
    test_set = DataLoaderSegmentation(folder_path_test_image, folder_path_test_masks)
    val_set = DataLoaderSegmentation(folder_path_val_image, folder_path_val_masks)

    # Init data loader
    train_loader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=5, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=True, num_workers=0)

    total_images = len(train_set) + len(test_set) + len(val_set)
    print("Load done:")
    print("Number of training images: {}, {}% of the dataset".format(len(train_set),round(100*len(train_set)/total_images,2)))
    print("Number of validation images: {}, {}% of the dataset".format(len(val_set),round(100*len(val_set)/total_images,2)))
    print("Number of testing images: {}, {}% of the dataset".format(len(test_set),round(100*len(test_set)/total_images,2)))

    num_epochs = (250)
    print("\nNumber of epochs: ",num_epochs)

    model = UNet(3,1,False).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # Linear scheduler: every 50 epochs the learning rate is multiplied by 0.8.
    al_param=50
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, al_param, gamma=0.8, last_epoch=-1, verbose=False)

    model_name = "model_V0"

    # Train model
    history_train_loss, history_val_loss, history_train_iou, history_val_iou = training_model(train_loader,loss_function,optimizer,model,num_epochs,model_name,scheduler,val_loader)

    # Save the model
    torch.save(model.state_dict(), 'model/'+ model_name + "_" + str(num_epochs)+ "epochs" + '.pt')

    # Save plot results
    plot_train_val(history_train_loss, history_val_loss,'loss')
    plot_train_val(history_train_iou,history_val_iou,'dice')