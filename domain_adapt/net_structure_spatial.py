from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
from skimage.transform import resize
import cv2
import torchvision.utils as vutils


class EncoderSourse(nn.Module):
    def __init__(self, ngpu=1):
        super(EncoderSourse, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=0),   # 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), #32 

            nn.Conv2d(64, 32, 5, stride=1, padding=0),  # 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            # nn.MaxPool2d(2, stride=2),  # 16

            nn.Conv2d(32, 16, 5, stride=1, padding=0),  # 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # 8         
        )

        self.toPIL = transforms.ToPILImage()
        self.reshape = transforms.Resize((60, 60))
        self.toTensor = transforms.ToTensor()

        self.softmax = nn.Softmax()

        num_rows, num_cols, num_fp = 109, 109, 16
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = torch.from_numpy(x_map).cuda()
        y_map = torch.from_numpy(y_map).cuda()

        self.x_map = x_map.view(num_rows * num_cols) #tf.reshape(x_map, [num_rows * num_cols])
        self.y_map = y_map.view(num_rows * num_cols)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_fp = num_fp
        self.batch_size = 16


    def spatial_softmax(self, x):
        # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
        features =  x.view(self.batch_size*self.num_fp, self.num_rows*self.num_cols)
        softmax = self.softmax( features )

        fp_x = torch.sum(torch.mul(self.x_map, softmax), 1).view(self.batch_size, self.num_fp)
        fp_y = torch.sum(torch.mul(self.y_map, softmax), 1).view(self.batch_size, self.num_fp)

        fp = torch.cat((fp_x, fp_y), 1)#.view(16, self.num_fp*2) #tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])
        return fp #(torch.matmul( softmax_out, self.fe_y ) ) + (torch.matmul( softmax_out, self.fe_x ) )

    def forward(self, input):
        output = self.main(input)
        key_points = self.spatial_softmax(output)

        img_cpu = input.cpu()
        key_cpu = key_points.cpu().detach().numpy()
        img_label = img_cpu.detach().numpy()
        img_label = resize(img_label, (16, 3, 60, 60))

        img_feature = img_label.copy() #resize(img_label, (16, 3, 109, 109))
        # img_feature = (img_feature + 1)*255
        # img_feature = img_feature.astype(np.uint8)

        for i in range(16):
            for j in range(16):
                row, col = key_cpu[i, j], key_cpu[i,j+16]
                row = int((row + 0.5)*60)
                col = int((col + 0.5)*60)
                # print(row, col)
                # if row >= 60 or col >= 60:
                #     continue
                img_feature[i,0,row, col] = 1
                img_feature[i,1,row, col] = 0
                img_feature[i,2,row, col] = 0
                # print(img_feature[i].dtype)
                # img_feature[i] = cv2.circle(img_feature[i],(row,col), 3, (1,0,0), -1)

        label = torch.from_numpy(img_label)
        feature = torch.from_numpy(img_feature)
        # img_cpu = input.cpu()
        # label = torch.empty((16, 3, 60, 60))
        # for i in range(16):
        #     img_pil = self.toPIL(img_cpu[i])
        #     img_reshape = self.reshape(img_pil)
        #     label[i] = self.toTensor(img_reshape)

        return key_points, feature.cuda(), label.cuda(), output # feature #output.view(-1, 1).squeeze(1)

class EncoderTarget(nn.Module):
    def __init__(self, ngpu=1):
        super(EncoderTarget, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=0),   # 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), #32 

            nn.Conv2d(64, 32, 5, stride=1, padding=0),  # 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            # nn.MaxPool2d(2, stride=2),  # 16

            nn.Conv2d(32, 16, 5, stride=1, padding=0),  # 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # 8         
        )
        self.softmax = nn.Softmax()

        num_rows, num_cols, num_fp = 109, 109, 16
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = torch.from_numpy(x_map).cuda()
        y_map = torch.from_numpy(y_map).cuda()

        self.x_map = x_map.view(num_rows * num_cols) #tf.reshape(x_map, [num_rows * num_cols])
        self.y_map = y_map.view(num_rows * num_cols)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_fp = num_fp
        self.batch_size = 16


    def spatial_softmax(self, x):
        # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
        features =  x.view(self.batch_size*self.num_fp, self.num_rows*self.num_cols)
        softmax = self.softmax( features )

        fp_x = torch.sum(torch.mul(self.x_map, softmax), 1).view(self.batch_size, self.num_fp)
        fp_y = torch.sum(torch.mul(self.y_map, softmax), 1).view(self.batch_size, self.num_fp)

        fp = torch.cat((fp_x, fp_y), 1)#.view(16, self.num_fp*2) #tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])
        return fp #(torch.matmul( softmax_out, self.fe_y ) ) + (torch.matmul( softmax_out, self.fe_x ) )

    def forward(self, input):
        output = self.main(input)
        key_points = self.spatial_softmax(output)

        img_cpu = input.cpu()
        key_cpu = key_points.cpu().detach().numpy()
        img_label = img_cpu.detach().numpy()
        img_label = resize(img_label, (16, 3, 60, 60))

        img_feature = img_label.copy() #resize(img_label, (16, 3, 109, 109))

        for i in range(16):
            for j in range(16):
                row, col = key_cpu[i, j], key_cpu[i,j+16]
                row = int((row + 0.5)*60)
                col = int((col + 0.5)*60)

                img_feature[i,0,row, col] = 1
                img_feature[i,1,row, col] = 0
                img_feature[i,2,row, col] = 0

        label = torch.from_numpy(img_label)
        feature = torch.from_numpy(img_feature)

        return key_points, feature.cuda(), label.cuda(), output # feature #output.view(-1, 1).squeeze(1)

class DecoderTarget(nn.Module):
    def __init__(self, ngpu=1):
        super(DecoderTarget, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True), 
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),   
            nn.Linear(512, 60*60),
            nn.Tanh()
                                   
            # nn.ConvTranspose2d(16, 32, 3, stride=2),  # 17
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),  # 33
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),  # 64
            # nn.Tanh()
        )

    def forward(self, input):
        feature = self.main(input)
        output = feature.view(16, 1, 60, 60)
        return output

class Decoder(nn.Module):
    def __init__(self, ngpu=1):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.Linear(32, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(True), 
            # nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(True),   
            nn.Linear(32, 60*60),
            nn.Tanh()
                                   
            # nn.ConvTranspose2d(16, 32, 3, stride=2),  # 17
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),  # 33
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),  # 64
            # nn.Tanh()
        )

    def forward(self, input):
        feature = self.main(input)
        output = feature.view(16, 1, 60, 60)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(True), 

            # nn.Linear(32, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),   

            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(True),  

            # nn.Linear(32, 1),
            # nn.Sigmoid()

            # input is (nc) x 8 x 8
            nn.Conv2d(16, 32, 5, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2), #32 
            # state size. (ndf) x 8 x 8
            nn.Conv2d(32, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2, stride=2), #32 
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(64, 32, 3, 1, 0, bias=False),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2, stride=2), #32 
            # # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(128, 1, 5, 1, 0, bias=False),

            # nn.Sigmoid()         
        )
        self.fc1 = nn.Linear(32*9*9, 1024)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        # print(output.shape)
        
        feature = output.view(16, 32*9*9)
        fc1 = self.act(self.fc1(feature))
        fc2 = self.fc2(fc1)
        out = self.sigmoid(fc2)

        # print(out.shape)
        return out.view(-1, 1).squeeze(1)

# netG = Generator(ngpu).to(device)
# netG.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)