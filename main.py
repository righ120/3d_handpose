import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import sys

import numpy as np
from data.data_loader import HandSegDataset, PoseNetDataset, PriorPoseDataset, RHD_v2, RandomCrop, CenteredCrop, TransposeAndToTensor

from models.HandSegNet import train_model as handsegnet_train, HandSegNet
from models.PoseNet import train_model as posenet_train, PoseNet
from models.PosePrior import train_model as poseprior_train, PosePrior


import skimage.transform as transform
import time
import copy


def crop_from_seg(imgs, masks):
    masks = masks.type(torch.bool)

    b, c, h, w = imgs.shape
    x_range = torch.arange(w).unsqueeze(1)
    y_range = torch.arange(h).unsqueeze(0)
    X = x_range.repeat(1, h).type(torch.int32)
    Y = y_range.repeat(w, 1).type(torch.int32)
    
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        X_masked = X[mask]
        x_min = min(X_masked) if len(X_masked) > 0 else 0
        x_max = max(X_masked)  if len(X_masked) > 0 else 0
        Y_masked = Y[mask]
        y_min = min(Y_masked) if len(Y_masked) > 0 else 0
        y_max = max(Y_masked)  if len(Y_masked) > 0 else 0

        
        cropsize_x = (x_max - x_min) if (x_max - x_min) > 50 else 50
        cropsize_y = (y_max - y_min) if (y_max - y_min) > 50 else 50
        cropsize = max(cropsize_x, cropsize_y)
        img = img[:,x_min:x_min + cropsize,y_min:cropsize].unsqueeze(0)
        img = F.interpolate(img, 256).squeeze()
        imgs[i] = img
   
    return imgs

def inference3d(handseg_model, posenet_model, poseprior_model, device, dataloaders, num_epochs=25):
     
    handseg_model.eval()
    posenet_model.eval()
    poseprior_model.eval()
   
    results = []
    # Iterate over data.
    for img,hand_side in dataloaders:
        
        img = img.to(device).type(torch.float32)
        hand_side = hand_side.to(device).type(torch.float32)

        outputs = handseg_model(img)
        _, preds = torch.max(outputs, 1)
        cropped_img = crop_from_seg(img, preds)
        score1, score2, score3 = posenet_model(cropped_img)
        coord3d, R = poseprior_model(((score1 + score2 + score3)/3,hand_side))
        results.append((coord3d, R))

    return results

if __name__ == "__main__":
     # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    handseg_model = HandSegNet().to(device)
    handseg_criterion = nn.CrossEntropyLoss().to(device)
    handseg_optimizer = torch.optim.Adam(handseg_model.parameters(), lr=0.0001)


    posenet_model = PoseNet().to(device)
    posenet_criterion = nn.MSELoss().to(device)
    posenet_optimizer = torch.optim.Adam(posenet_model.parameters(), lr=0.0001)

    poseprior_model = PosePrior().to(device)
    poseprior_criterion = nn.MSELoss().to(device)
    poseprior_optimizer = torch.optim.Adam(poseprior_model.parameters(), lr=0.0001)
    

    handseg_transform = transforms.Compose([
        RandomCrop(256),
        TransposeAndToTensor()
        ])

    posenet_transform = transforms.Compose([
        CenteredCrop(256)
        ])

    # train handsegnet
    train_dataset = HandSegDataset("./data/RHD_published_v2/training","training", handseg_transform)
    valid_dataset = HandSegDataset("./data/RHD_published_v2/evaluation","evaluation",handseg_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    vaild_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)

    handseg_model = handsegnet_train(handseg_model, handseg_criterion, handseg_optimizer, device,
                dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)


    # train posenet
    train_dataset = PoseNetDataset("./data/RHD_published_v2/training","training", posenet_transform=posenet_transform)
    valid_dataset = PoseNetDataset("./data/RHD_published_v2/evaluation","evaluation", posenet_transform=posenet_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    vaild_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)
    
    posenet_model = posenet_train(posenet_model, posenet_criterion, posenet_optimizer, device,
                dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)
    

    # train poseprior
    train_dataset = PriorPoseDataset("./data/RHD_published_v2/training","training", posenet_transform=posenet_transform)
    valid_dataset = PriorPoseDataset("./data/RHD_published_v2/evaluation","evaluation", posenet_transform=posenet_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    vaild_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)

    poseprior_model = poseprior_train(posenet_model, poseprior_model, poseprior_criterion, poseprior_optimizer, device,
                dataloaders={"train":train_data_loader, "valid":vaild_data_loader}, scheduler=None, num_epochs=25)


    # infer 3D coords
    device = torch.device('cpu')
    inf_dataset = RHD_v2("./data/RHD_published_v2/evaluation","evaluation", handseg_transform)

    inf_data_loader = torch.utils.data.DataLoader(inf_dataset, batch_size=2, shuffle=False)

    handseg_model.to(device)
    posenet_model.to(device)
    poseprior_model.to(device)

    results = inference3d(handseg_model, posenet_model, poseprior_model, device,  dataloaders=inf_data_loader, num_epochs=25)