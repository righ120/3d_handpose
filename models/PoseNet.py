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

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
        self.layer2 = nn.Conv2d(64, 64, (3,3), stride=1, padding=1)
        self.layer3 = nn.MaxPool2d((4,4), stride=2, padding=1)

        self.layer4 = nn.Conv2d(64, 128, (3,3), stride=1, padding=1)
        self.layer5 = nn.Conv2d(128, 128, (3,3), stride=1, padding=1)
        self.layer6 = nn.MaxPool2d((4,4), stride=2, padding=1)

        self.layer7 = nn.Conv2d(128, 256, (3,3), stride=1, padding=1)
        self.layer8 = nn.Conv2d(256, 256, (3,3), stride=1, padding=1)
        self.layer9 = nn.Conv2d(256, 256, (3,3), stride=1, padding=1)
        self.layer10 = nn.Conv2d(256, 256, (3,3), stride=1, padding=1)
        self.layer11 = nn.MaxPool2d((4,4), stride=2, padding=1)

        self.layer12 = nn.Conv2d(256, 512, (3,3), stride=1, padding=1)
        self.layer13 = nn.Conv2d(512, 512, (3,3), stride=1, padding=1)
        self.layer14 = nn.Conv2d(512, 512, (3,3), stride=1, padding=1)
        self.layer15 = nn.Conv2d(512, 512, (3,3), stride=1, padding=1)
        self.layer16 = nn.Conv2d(512, 512, (3,3), stride=1, padding=1)
        self.layer17 = nn.Conv2d(512, 21, (1,1), stride=1)
        
        

        self.layer19 = nn.Conv2d(533, 128, (7,7), stride=1, padding=3)
        self.layer20 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer21 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer22 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer23 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer24 = nn.Conv2d(128, 21, (1,1), stride=1)

        self.layer26 = nn.Conv2d(554, 128, (7,7), stride=1, padding=3)
        self.layer27 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer28 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer29 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer30 = nn.Conv2d(128, 128, (7,7), stride=1, padding=3)
        self.layer31 = nn.Conv2d(128, 21, (1,1), stride=1)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer3(F.relu(self.layer2(x)))
        
        x = F.relu(self.layer4(x))
        x = self.layer6(F.relu(self.layer5(x)))
        
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = self.layer11(F.relu(self.layer10(x)))

        x = F.relu(self.layer12(x))
        x = F.relu(self.layer13(x))
        x = F.relu(self.layer14(x))
        x = F.relu(self.layer15(x))
        x = F.relu(self.layer16(x))
        score1 = self.layer17(x)
        
        cat1 = torch.cat((x, score1),1)
        x = F.relu(self.layer19(cat1))
        x = F.relu(self.layer20(x))
        x = F.relu(self.layer21(x))
        x = F.relu(self.layer22(x))
        x = F.relu(self.layer23(x))
        score2 = self.layer24(x)
        
        cat2 = torch.cat((cat1, score2),1)
        x = F.relu(self.layer26(cat2))
        x = F.relu(self.layer27(x))
        x = F.relu(self.layer28(x))
        x = F.relu(self.layer29(x))
        x = F.relu(self.layer30(x))
        score3 = self.layer31(x)

        return score1, score2, score3


def train_model(model, criterion, optimizer, device, dataloaders, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    iter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs,labels in dataloaders[phase]:
                
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

    model = PoseNet().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    data_transform = transforms.Compose([
        CenteredCrop(256)
        ])

    train_dataset = PoseNetDataset("../data/RHD_published_v2/training","training", posenet_transform=data_transform)
    valid_dataset = PoseNetDataset("../data/RHD_published_v2/evaluation","evaluation", posenet_transform=data_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    vaild_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)

    train_model(model, criterion, optimizer, device, dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)