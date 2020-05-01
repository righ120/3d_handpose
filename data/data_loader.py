
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt

import os
from skimage import io, transform
import pickle

import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HandSegDataset(data.Dataset):
    """3D Handpose Dataset"""

    def __init__(self, root_dir, phase, handseg_transform=None, posenet_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): . (train/test)
            transform ():
        """

        # load annotations of this set
        annos = []
        with open(os.path.join(root_dir, 'anno_%s.pickle' % phase), 'rb') as fi:
            anno_all = pickle.load(fi)
            for id, anno in anno_all.items():
                annos.append((id, anno))
        self.annos = annos
        self.root_dir = root_dir
        self.phase = phase
        self.handseg_transform = handseg_transform
        self.posenet_transform = posenet_transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):

        id, mat = self.annos[index]
        # load data
        img = io.imread(os.path.join(self.root_dir, 'color', '%.5d.png' % id))
        mask = io.imread(os.path.join(self.root_dir, 'mask', '%.5d.png' % id))
        

        if self.handseg_transform:
            img, mask = self.handseg_transform((img, mask))
            one_map = torch.ones_like(mask)
            cond_l = torch.gt(mask, one_map)
            mask[~cond_l] = 0
            mask[cond_l] = 1

        return img, mask


class PoseNetDataset(data.Dataset):
    """3D Handpose Dataset"""

    def __init__(self, root_dir, phase, handseg_transform=None, posenet_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): . (train/test)
            transform ():
        """

        # load annotations of this set
        annos = []
        with open(os.path.join(root_dir, 'anno_%s.pickle' % phase), 'rb') as fi:
            anno_all = pickle.load(fi)
            for id, anno in anno_all.items():
                annos.append((id, anno))
        self.annos = annos
        self.root_dir = root_dir
        self.phase = phase
        self.handseg_transform = handseg_transform
        self.posenet_transform = posenet_transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):


        id, mat = self.annos[index]
        
        # load data
        img = io.imread(os.path.join(self.root_dir, 'color', '%.5d.png' % id))
        mask = io.imread(os.path.join(self.root_dir, 'mask', '%.5d.png' % id))
        

        # get info from annotation dictionary
        kp_coord_uv = mat['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (mat['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
        kp_coord_xyz = mat['xyz']  # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = mat['K']  # matrix containing intrinsic parameters


        _, mask = ToTensor()((img, mask))
        one_map = torch.ones_like(mask)
        cond_l = torch.gt(mask, one_map)
        cond_r = torch.gt(mask, one_map*17)
        left = False
        if len(mask[cond_l]) > len(mask[cond_r]):
            hand_side = torch.Tensor([1, 0])
            keypoints = torch.Tensor(kp_coord_uv[:21, :])
            keypoints_vis = kp_visible[:21]
            if sum(keypoints_vis) != 0:
                left = True
        if not left:
            hand_side = torch.Tensor([0, 1])
            keypoints = torch.Tensor(kp_coord_uv[21:, :])
            keypoints_vis = kp_visible[21:]
        
        #print(len(mask[cond_l]),len(mask[cond_r]))
        #crop_center = keypoints[12]
        crop_center = keypoints[12].flip(0)
        print(kp_visible)
        
        # select visible coords only
        kp_coord_h = keypoints[keypoints_vis, 1]
        kp_coord_w = keypoints[keypoints_vis, 0]
        kp_coord_hw = torch.stack([kp_coord_h, kp_coord_w], 1)
       
        print(crop_center)
        # determine size of crop (measure spatial extend of hw coords first)
        min_coord = torch.min(kp_coord_hw,0).values
        max_coord = torch.max(kp_coord_hw,0).values
        crop_size = 2 * torch.max(crop_center  - min_coord, max_coord - crop_center)
        crop_size = torch.max(crop_size)
        print(crop_size)
        # calculate necessary scaling

        scale = 256.0 / crop_size
        #print(scale)
        #scale = scale.clamp(min=1, max=10)

        cropped_img = self.posenet_transform((img, crop_center, crop_size))

        kp_coord_uv21_u = (keypoints[:,1] - crop_center[0] + crop_size/2) * scale 
        kp_coord_uv21_v = (keypoints[:,0] - crop_center[1] + crop_size/2) * scale
        keypoint_uv21 = torch.stack([kp_coord_uv21_u, kp_coord_uv21_v], 1)
        keypoint_hw = torch.stack([kp_coord_uv21_v, kp_coord_uv21_u], 1)
        #print(keypoint_uv21)
        score_map = ScoreMap(256)((keypoint_uv21,keypoints_vis))

        return cropped_img, score_map


class RHD_v2(data.Dataset):
    """3D Handpose Dataset"""

    def __init__(self, root_dir, phase, handseg_transform=None, posenet_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): . (train/test)
            transform ():
        """

        # load annotations of this set
        annos = []
        with open(os.path.join(root_dir, 'anno_%s.pickle' % phase), 'rb') as fi:
            anno_all = pickle.load(fi)
            for id, anno in anno_all.items():
                annos.append((id, anno))
        self.annos = annos
        self.root_dir = root_dir
        self.phase = phase
        self.handseg_transform = handseg_transform
        self.posenet_transform = posenet_transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):


        id, mat = self.annos[index]
        # load data
        img = io.imread(os.path.join(self.root_dir, 'color', '%.5d.png' % id))
        mask = io.imread(os.path.join(self.root_dir, 'mask', '%.5d.png' % id))
        #depth = io.imread(os.path.join(self.root_dir, 'depth', '%.5d.png' % id))
        
        # get info from annotation dictionary
        kp_coord_uv = mat['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (mat['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
        kp_coord_xyz = mat['xyz']  # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = mat['K']  # matrix containing intrinsic parameters

        # Project world coordinates into the camera frame
        kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
        kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]


        if self.handseg_transform:
            img, mask = self.handseg_transform((img, mask))
            one_map = torch.ones_like(mask)
            cond_l = torch.gt(mask, one_map)
            mask[~cond_l] = 0
            mask[cond_l] = 1

        if self.posenet_transform:
            if self.phase == "training":
                cropped_img = self.hand_transform(img, xy, crop_size, scale=1)
            elif self.phase == "evaluation":
                cropped_img = self.hand_transform(img, xy, crop_size, scale=1)
        else:
            cropped_img = None
        return img, mask, cropped_img



class ScoreMap(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        coords_uv, coords_uv_vis = x
        h, w = self.output_size
        

        # create meshgrid
        x_range = torch.arange(w).unsqueeze(1)
        y_range = torch.arange(h).unsqueeze(0)

        X = x_range.repeat(1, h).type(torch.float32).unsqueeze(-1)
        Y = y_range.repeat(w, 1).type(torch.float32).unsqueeze(-1)
       
        X = X.repeat([1, 1, 21])
        Y = Y.repeat([1, 1, 21])
        
        X -= coords_uv[:, 0]
        Y -= coords_uv[:, 1]

        dist = X**2 + Y**2
        
        scoremap = np.array(np.exp(-dist / 25**2) * coords_uv_vis).transpose((2,0,1))
        
        #return torch.from_numpy(scoremap).type(torch.float32)
        return scoremap

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        image, mask = x

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]

        return image, mask


class CenteredCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, input):
        img, crop_center, crop_size = input
        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        top = (crop_center[0] - crop_size/2).floor().type(torch.int32)
        left = (crop_center[1] - crop_size/2).floor().type(torch.int32)
        image = img[top: top + crop_size.type(torch.int32),
                      left: left + crop_size.type(torch.int32)]

        image = transform.resize(image, (new_h, new_w))
        #image = np.array(image).transpose((2, 0, 1))

        return torch.from_numpy(image)

class HueAug(object):
    def __call__(self, x):
        image, mask = x
        image = transforms.ColorJitter(hue=0.1)(image)
        return image, mask

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, x):
        image, mask = x

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        mask = transform.resize(mask, (new_h, new_w))

        return image, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, x):
        image, mask = x

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).transpose((2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(np.array(mask))

if __name__ == "__main__":
    
    fig = plt.figure()
    data_transform = transforms.Compose([
        CenteredCrop(256)
        ])
 
    dataset = PoseNetDataset(root_dir="RHD_published_v2\\training", phase="training", posenet_transform=data_transform)
    #dataset = RHD_v2(root_dir="RHD_published_v2\\evaluation", phase="evaluation", handseg_transform=data_transform)
    
    for i in range(281, len(dataset)):
        
        cropped_img, score_map = dataset[i]
        #print(kp)
        fig = plt.figure(1)
        ax1 = fig.add_subplot('221')
        ax2 = fig.add_subplot('222')
        ax3 = fig.add_subplot('223')
        ax4 = fig.add_subplot('224')
        #ax1.imshow(img)
        #ax1.plot(kp_coord_hw[:, 0], kp_coord_hw[:, 1], 'ro')
        #ax1.plot(kp_coord_uv_proj[kp_visible, 0], kp_coord_uv_proj[kp_visible, 1], 'gx')
        ax1.imshow(cropped_img)
        #ax1.plot(kp_coord_uv[:, 0], kp_coord_uv[:, 1], 'ro')
        ax2.imshow(score_map[0, :,:])
        ax3.imshow(score_map[1, :,:])
        ax4.imshow(score_map[2, :,:])
        plt.show()
        
        
        
        