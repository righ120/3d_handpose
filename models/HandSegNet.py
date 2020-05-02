import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data.data_loader import HandSegDataset, RandomCrop,  TransposeAndToTensor
import time
import copy



class HandSegNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.layer17 = nn.Conv2d(512, 2, (1,1), stride=1, padding=0)
        
        self.layer18 = nn.Upsample(size=(256, 256), mode="bilinear")


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
        x = self.layer17(F.relu(self.layer16(x)))
        
        x = self.layer18(x)
        
        return x

def train_model(model, criterion, optimizer, device, dataloaders, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.int64)
                #print(labels[0])
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print(outputs.shape, labels.shape)
                    #print(criterion)
                    loss = criterion(outputs, labels)
                    #print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    model = HandSegNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    data_transform = transforms.Compose([
        RandomCrop(256),
        TransposeAndToTensor()
        ])
    
    train_dataset = HandSegDataset("../data/RHD_published_v2/training","training", data_transform)
    valid_dataset = HandSegDataset("../data/RHD_published_v2/evaluation","evaluation")

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    vaild_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)

    train_model(model, criterion, optimizer, device,
                dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)