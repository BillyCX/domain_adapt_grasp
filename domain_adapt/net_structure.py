from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

class EncoderSourse(nn.Module):
    def __init__(self, ngpu=1):
        super(EncoderSourse, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),   # 64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #32 

            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),  # 16

            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 8         
        )

    def forward(self, input):
        output = self.main(input)
        return output #index.view(16, 16).float()/(16*16), feature # feature #output.view(-1, 1).squeeze(1)

class EncoderTarget(nn.Module):
    def __init__(self, ngpu=1):
        super(EncoderTarget, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.Conv2d(3, 16, 7, stride=2, padding=0),   # 128
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), # 64

            nn.Conv2d(3, 32, 3, stride=1, padding=1),   # 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #32 

            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),  # 16

            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # 8
        )

    def forward(self, input):
        output = self.main(input)
        return output 

class Decoder(nn.Module):
    def __init__(self, ngpu=1):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2),  # 17
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),  # 33
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),  # 64
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        # output = feature.view(16, 1, 64, 64)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 8 x 8
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()         
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# netG = Generator(ngpu).to(device)
# netG.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)