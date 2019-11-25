from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

# last_conv_size = 53
last_conv_size = 28

class CNNLatentDistrib(nn.Module):
    def __init__(self, latent_n=3, last_conv_channel=16):
        super(CNNLatentDistrib, self).__init__()
        self.cnn2_size = 28
        self.cnn2_channel = last_conv_channel
        self.cnn2 = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=2, padding=0),   # 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), #32 

            nn.Conv2d(128, 64, 3, stride=1, padding=0),  # 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),  # 16

            nn.Conv2d(64, last_conv_channel, 3, stride=1, padding=0),  # 16
            nn.BatchNorm2d(last_conv_channel),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # 8      

            # nn.Conv2d(64, last_conv_channel, 3, stride=1, padding=0),  # 16
            # nn.BatchNorm2d(last_conv_channel),
            # nn.ReLU(True),
            # # nn.MaxPool2d(2, stride=2),  # 8                    
        )

        self.fc1 = nn.Linear(self.cnn2_channel*self.cnn2_size*self.cnn2_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, latent_n)
        self.fc4 = nn.Linear(128, latent_n)
        self.actf = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        self.last_conv_channel = last_conv_channel

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().float().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).float().cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        fc1 = self.actf(self.fc1(h))
        fc2 = self.actf(self.fc2(fc1))

        mu, logvar = self.fc3(fc2), self.fc4(fc2)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def get_mu_conv(self, input):
        h = self.cnn2(input)
     
        z, mu, logvar = self.bottleneck(h.view(-1, self.cnn2_channel*self.cnn2_size*self.cnn2_size))

        return mu, h

    def get_mu_std(self, input):
        h = self.cnn2(input)

        z, mu, logvar = self.bottleneck(h.view(-1, self.cnn2_channel*self.cnn2_size*self.cnn2_size))

        return mu, logvar

    def forward(self, input):
        h = self.cnn2(input)
        print(h.shape)
        z, mu, logvar = self.bottleneck(h.view(-1, self.cnn2_channel*self.cnn2_size*self.cnn2_size))

        return z, h


class CNNTarget(nn.Module):
    def __init__(self, latent_n=3, last_conv_channel=16):
        super(CNNTarget, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=0),   # 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), #32 

            nn.Conv2d(64, 32, 5, stride=2, padding=0),  # 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            # nn.MaxPool2d(2, stride=2),  # 16

            nn.Conv2d(32, 32, 3, stride=1, padding=0),  # 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # 8      

            nn.Conv2d(32, last_conv_channel, 3, stride=1, padding=0),  # 16
            nn.BatchNorm2d(last_conv_channel),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # 8                    
        )
        # self.fc1 = nn.Linear(last_conv_channel*last_conv_size*last_conv_size, 1024)
        # self.fc1_bn = nn.BatchNorm1d(1024)

        # self.fc2 = nn.Linear(1024, 2048)
        # self.fc2_bn = nn.BatchNorm1d(2048)

        # self.fc3 = nn.Linear(2048, last_conv_channel*last_conv_size*last_conv_size)

        # self.actf = nn.ReLU()
        self.last_conv_channel = last_conv_channel

    def forward(self, input):
        h = self.main(input)
        # print(h.shape)
        # fc_input = h.view(-1, self.last_conv_channel*last_conv_size*last_conv_size)
        # fc1 = self.fc1_bn(self.actf(self.fc1(fc_input)))
        # fc2 = self.fc2_bn(self.actf(self.fc2(fc1)))
        # fc3 = self.fc3(fc2)

        return h, h #fc3.view(-1, self.last_conv_channel, last_conv_size, last_conv_size)        

class CNNSpatial(nn.Module):
    def __init__(self, ngpu=1):
        super(CNNSpatial, self).__init__()
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

        # self.toPIL = transforms.ToPILImage()
        # self.reshape = transforms.Resize((60, 60))
        # self.toTensor = transforms.ToTensor()

        self.softmax = nn.Softmax()

        num_rows, num_cols, num_fp = 53, 53, 16
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

    def get_mu_conv(self, input):
        return self.forward(input)

    def forward(self, input):
        output = self.main(input)
        key_points = self.spatial_softmax(output)

        # # img_cpu = input.cpu()
        # # key_cpu = key_points.cpu().detach().numpy()
        # # img_label = img_cpu.detach().numpy()
        # # img_label = resize(img_label, (16, 3, 60, 60))

        # # img_feature = img_label.copy() #resize(img_label, (16, 3, 109, 109))
        # # img_feature = (img_feature + 1)*255
        # # img_feature = img_feature.astype(np.uint8)

        # for i in range(16):
        #     for j in range(16):
        #         row, col = key_cpu[i, j], key_cpu[i,j+16]
        #         row = int((row + 0.5)*60)
        #         col = int((col + 0.5)*60)
        #         # print(row, col)
        #         # if row >= 60 or col >= 60:
        #         #     continue
        #         img_feature[i,0,row, col] = 1
        #         img_feature[i,1,row, col] = 0
        #         img_feature[i,2,row, col] = 0
        #         # print(img_feature[i].dtype)
        #         # img_feature[i] = cv2.circle(img_feature[i],(row,col), 3, (1,0,0), -1)

        # label = torch.from_numpy(img_label)
        # feature = torch.from_numpy(img_feature)
        # # img_cpu = input.cpu()
        # # label = torch.empty((16, 3, 60, 60))
        # # for i in range(16):
        # #     img_pil = self.toPIL(img_cpu[i])
        # #     img_reshape = self.reshape(img_pil)
        # #     label[i] = self.toTensor(img_reshape)

        return key_points, output

class FCDecoder(nn.Module):
    def __init__(self, input_length):
        super(FCDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True), 

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),  
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),  

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),  

            nn.Linear(1024, 1*128*128),
            nn.Tanh()
        )
        # self.fc1 = nn.Linear(n_z, 128*32*32)   

    def forward(self, input):
        # h = self.fc1(input)
        # output = self.main(h.view(-1, 16, 8, 8))
        output = self.main(input)
        return output.view(-1, 1, 128, 128)


class CnnDecoder(nn.Module):
    def __init__(self, input_length):
        super(CnnDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 7, stride=1),  # 59
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 5, stride=1),  # 63
            nn.ReLU(True),            
            # nn.ConvTranspose2d(16, 8, 3, stride=2),  # 63
            # nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 4, stride=2, padding=0),  # 128
            nn.Tanh()
        )
        # self.fc1 = nn.Linear(n_z, 128*32*32)   

    def forward(self, input):
        # h = self.fc1(input)
        # output = self.main(h.view(-1, 16, 8, 8))
        # print(input.shape)

        output = self.main(input)
        # print(output.shape)
        return output.view(-1, 1, 128, 128)


class CnnBinaryDiscriminator(nn.Module):
    def __init__(self, input_channel):
        super(CnnBinaryDiscriminator, self).__init__()
        self.cnn_size = 9
        self.main = nn.Sequential(           
            # input is (nc) x 8 x 8
            nn.Conv2d(input_channel, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2, stride=2),

            # state size. (ndf) x 8 x 8
            nn.Conv2d(64, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2, stride=2),

            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(32, 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2, stride=2),
        )
        self.fc1 = nn.Linear(16*self.cnn_size*self.cnn_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, input):
        output = self.main(input)
        # print(output.shape)
        fc1 = self.relu(self.fc1(output.view(-1, 16*self.cnn_size*self.cnn_size)))
        fc2 = self.relu(self.fc2(fc1))
        fc3 = self.sigmoid(self.fc3(fc2))

        return fc3.view(-1, 1).squeeze(1), output[0].view(16, 1, self.cnn_size, self.cnn_size)

class CnnMultiCategoryDiscriminator(nn.Module):
    def __init__(self, input_channel, last_conv_size=16):
        super(CnnMultiCategoryDiscriminator, self).__init__()
        self.cnn_size = 6
        self.main = nn.Sequential(           
            # input is (nc) x 8 x 8
            nn.Conv2d(input_channel, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # state size. (ndf) x 8 x 8
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc1 = nn.Linear(16*self.cnn_size*self.cnn_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        # self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        output = self.main(input)
        # print(output.shape)
        fc1 = self.drop(self.relu(self.fc1(output.view(-1, 16*self.cnn_size*self.cnn_size))))
        fc2 = self.drop(self.relu(self.fc2(fc1)))
        fc3 = self.fc3(fc2)
        # print(fc3.shape)
        # return fc3.view(-1, 1).squeeze(1), output[0].view(16, 1, self.cnn_size, self.cnn_size)
        return fc3, output[0].view(16, 1, self.cnn_size, self.cnn_size)