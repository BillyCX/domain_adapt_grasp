from __future__ import print_function, division
import os
import glob
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.filters import threshold_otsu
from skimage import io
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

exp_type = 'pen'


class ExpDataProcessor():
    def __init__(self):
        img_transform = transforms.Compose([
                                    # EdgeCrop(200, 440, 200, 630),        
                                    # Rescale((60, 105)),
                                    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ToTensorNoLabel(),
                                    ])        
        self.transform = img_transform
        self.label_min = np.array([0.35, -0.2, 0.1])
        self.label_max = np.array([0.55, 0.2, 3.1])
        self.label_scale = np.array([0.5, 1, 1])

    def process(self, raw_img):
        # img = transform.resize(raw_img, (540, 960))
        # img = np.fliplr(img)
        img = raw_img

        # io.imsave('/home/xi/exp_adda_ori/' + str(time.time()) + '.png', img)

        sample = {'image': img}

        if self.transform:
            sample = self.transform(sample)

        return sample            

    def unnormalize_result(self, pred_latent):
        result = pred_latent/self.label_scale * (self.label_max-self.label_min) + self.label_min
        return result


class ObjectsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, img_type, transform=None, load_label=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list = []
        self.label_list = []
        self.img_type = img_type
        self.load_label = load_label

        if exp_type == 'pen':
            self.weights = [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.0, 1.3, 1.0, 1.0, 1.7, 3.0]
        elif exp_type == 'cup':
            self.weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 4.0, 1.0, 1.7, 1.0, 1.0, 1.7, 5]

        self.label_min = np.array([0.35, -0.2, 0.1])
        self.label_max = np.array([0.55, 0.2, 3.1])
        self.label_scale = np.array([0.5, 1, 1])

        dir_length = len(root_dir)
        print(root_dir)
        for path, subdirs, files in os.walk(root_dir):
            # for name in files:
            #     self.img_list.append(os.path.join(path, name))
            file_num = len(files)
            if file_num == 0 or path.find('img') == -1:
                continue 

            size = 32*50
            if img_type == 'obj':
                if exp_type == 'pen':
                    obj_id = int(path[17:-4]) - 1
                elif exp_type == 'cup':
                    obj_id = int(path[21:-4]) - 1
                size = int(size * self.weights[obj_id])
                print(obj_id, size)

            # file_list = np.random.randint(0, file_num, size=size)
            # for i in file_list:
            #     img_path = os.path.join(path, files[i])
            #     self.img_list.append(img_path)
            #     # print(path[:-3], files[i][:-4])
            #     if load_label:
            #         label_path = path[:-3] + 'label/' + files[i][:-4] + '.txt'
            #         self.label_list.append(label_path)
            #         # print(img_path)
            #         # print(label_path)
            #                 
            for f in files:
                img_path = os.path.join(path, f)                
                self.img_list.append(img_path)

        # self.img_list = glob.glob(root_dir + '*.png')
        print('dataset size:', len(self.img_list), self.img_list[0])

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        image = io.imread(img_name)

        if self.img_type == 'src' and np.random.rand() > 1:
            image_random = image.copy()
            rand_r = np.random.rand()
            rand_g = np.random.rand()
            rand_b = np.random.rand()
            image_random[:,:,0] = image[:,:,0] + rand_r * 250
            image_random[:,:,1] = image[:,:,1] + rand_g * 250
            image_random[:,:,2] = image[:,:,2] + rand_b * 250

            # image = image_random
            image = np.where(image<50, image_random, image)
            # print(image.min(), image.max())
        # image = transform.resize(image, (540, 960))

        # image = image[200:440,
        #               200:630]
        # # image = np.fliplr(image)
        # # print(img_name)
        # io.imsave('./test.png', image)

        # if img_name.find('src') != -1:
        #     thresh = threshold_otsu(image)
        #     image = image > thresh        

        if self.load_label:
            label_path = self.label_list[idx]
            file = open(label_path, 'r')
            label = file.read().split(',')
            label_numpy = np.array(label, dtype=float)
            label_norm = (label_numpy - self.label_min)/(self.label_max-self.label_min)
            label_norm *= self.label_scale
            # print (label_numpy)

            sample = {'image': image, 'label': label_norm}
        else:
            sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.
./dataset/noobj/
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        sample['image'] = img
        return sample


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

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        sample['image'] = image
        return sample

class EdgeCrop(object):
    """Crop the middle part of the image.

    Args:
        edge_size (int): size of the edge to cut off
    """

    def __init__(self, edge_top, edge_down, edge_left, edge_right):
        # assert isinstance(edge_size, int)
        self.edge_top = edge_top
        self.edge_down = edge_down
        self.edge_left = edge_left
        self.edge_right = edge_right

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        image = image[self.edge_top: self.edge_down,
                      self.edge_left: self.edge_right]

        # print(image.shape)

        sample['image'] = image
        return sample

class Normalize(object):
    """Normalize Tensors."""
    def __init__(self, mean, std):
        # assert isinstance(edge_size, int)
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image = (image - self.mean)/self.std
        sample['image'] = image
        return sample

class ToTensorNoLabel(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # ten = torch.from_numpy(image).float()
        # print('here', ten.type())

        return {'image': torch.from_numpy(image).float()}

class ToTensorWithLabel(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # ten = torch.from_numpy(image).float()
        # print('here', ten.type())

        return {'image': torch.from_numpy(image).float(), 'label': torch.from_numpy(label).float()}



def get_dataset(img_type, root_dir='./model_dataset/', load_label=False):
    if load_label:
        img_transform = transforms.Compose([
                                    EdgeCrop(200, 440, 200, 630),        
                                    Rescale((60, 105)),
                                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ToTensorWithLabel(),
                                    ])
    else:
        img_transform = transforms.Compose([
                                    EdgeCrop(200, 440, 200, 630),        
                                    Rescale((60, 105)),
                                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ToTensorNoLabel(),
                                    ])        

    objects_dataset = ObjectsDataset(img_type=img_type, root_dir=root_dir, transform=img_transform, load_label=load_label)

    # for i in range(len(objects_dataset)):
    #     sample = objects_dataset[i]['image']
    #     plt.imshow(sample)
    #     plt.pause(0.001)

    #     if i == 3:
    #         plt.show()
    #         break

    #     print(i, sample.shape, sample.max(), sample.min())

    return objects_dataset

# dataloader = DataLoader(objects_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched.size())

# for i in range(len(objects_dataset)):
#     sample = objects_dataset[i]

#     print(i, sample.shape, sample.max(), sample.min())

#     # plt.imshow(sample)
#     # plt.pause(0.001)

#     # if i == 3:
#     #     plt.show()
#     #     break