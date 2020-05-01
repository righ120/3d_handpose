
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data.data_loader import PoseNetDataset, CenteredCrop, ToTensor
import skimage.transform as transform
import time
import copy

class ViewPoints(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.layer1 = nn.Conv2d(21, 32, (3,3), stride=2, padding=1)
        self.layer2 = nn.Conv2d(32, 32, (3,3), stride=1, padding=1)
        self.layer3 = nn.Conv2d(32, 64, (3,3), stride=1, padding=1)
        self.layer4 = nn.Conv2d(64, 64, (3,3), stride=2, padding=1)
        self.layer5 = nn.Conv2d(64, 128, (3,3), stride=1, padding=1)
        self.layer6 = nn.Conv2d(128, 128, (3,3), stride=2, padding=1)

        self.layer8 = nn.Linear(130, 512)
        self.layer8_d = nn.Dropout(0.2)
        self.layer9 = nn.Linear(512, 512)
        self.layer9_d = nn.Dropout(0.2)
        self.layer10 = nn.Linear(512, P)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))

        x = torch.cat((x.view(), torch.Tensor([0, 1])))
        x = self.layer8_d(self.layer8(x))
        x = self.layer9_d(self.layer9(x))
        x = self.layer10(x)

        return x

class PosePrior(nn.Module):
    def __init__():
        super().__init__()
        self.viewpoints = ViewPoints(3)
        self.canonical_coord = ViewPoints(63)

    def forward(self, x):
        w = self.viewpoints(x)
        R = self.canonical_coord(x)
        return w, R

def train_model(posenet_model, poseprior_model, viewpoints_criterion, 
                    R_criterion, optimizer, device, dataloaders, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    iter = 0
    posenet_model.eval()
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
                    score1, score2, score3 = posenet_model(inputs)
                    coord3d, R = poseprior_model((score1 + score2 + score3)/3)
                    
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

    model = PoseNet().to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    second_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    third_optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)


    data_transform = transforms.Compose([
        CenteredCrop(256)
        ])

    train_dataset = PoseNetDataset("..\\data\\RHD_published_v2\\training","training", posenet_transform=data_transform)
    valid_dataset = PoseNetDataset("..\\data\\RHD_published_v2\\evaluation","evaluation")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False)
    vaild_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=2, shuffle=False)

    train_model(model, criterion, optimizer, device,
                dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)

