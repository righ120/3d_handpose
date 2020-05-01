

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import sys


from data.data_loader import PoseNetDataset, CenteredCrop, ToTensor
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.PoseNet import PoseNet
from models.PosePrior import PosePrior
from models.HandSegNet import HandSegNet

import skimage.transform as transform
import time
import copy

def train_model(handseg_model, posenet_model, poseprior_model, viewpoint_crt, R_crt, optimizer, device, dataloaders, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    iter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs,labels in dataloaders[phase]:
                #print(iter)
                #iter += 1    
                inputs = inputs.to(device).type(torch.float32)
                
                labels = labels.to(device).type(torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    score1, score2, score3 = model(inputs)
                    score1 = F.interpolate(score1, 256)
                    score2 = F.interpolate(score2, 256)
                    score3 = F.interpolate(score3, 256)
                    loss = criterion(score1, labels)
                    loss += criterion(score2, labels)
                    loss += criterion(score3, labels)
                    loss /= 3
                    #print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            #if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
     # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    handseg_model = HandSegNet().to(device)
    posenet_model = PoseNet().to(device)
    poseprior_model = PosePrior().to(device)

    seg_loss = nn.CrossEntropyLoss().to(device)
    scoremap_loss = nn.MSELoss().to(device)
    viewpoints_criterion = nn.MSELoss().to(device)
    R_criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    second_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    third_optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    
    handseg_transform = transforms.Compose([
        RandomCrop(256),
        ToTensor()
        ])

    posenet_transform = transforms.Compose([
        CenteredCrop(256)
        ])

    train_dataset = PoseNetDataset(".\\data\\RHD_published_v2\\training","training", handseg_transform=handseg_transform,posenet_transform=posenet_transform)
    valid_dataset = PoseNetDataset(".\\data\\RHD_published_v2\\evaluation","evaluation")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True)
    vaild_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=2, shuffle=False)

    train_model(handseg_model, posenet_model, poseprior_model, criterion1, optimizer, device,
                dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)