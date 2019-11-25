from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import data_loader as training_data
from algos.domain_adapt_vae_multi_category import MultiCategoryVAE

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width osf the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--train_model', type=str, help='model name to train')
parser.add_argument('--method', type=str, help='model name to train')

opt = parser.parse_args()
print(opt)

try:
    path = 'results/%s/%s/' % (opt.outf, opt.method)
    os.makedirs(path)
except OSError:
    pass


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


data_obj = training_data.get_object_dataset(root_dir='./dataset/obj/')
data_noobj = training_data.get_object_dataset(root_dir='./dataset/noobj/')
data_src = training_data.get_sourse_dataset(root_dir='./dataset/src/')

loader_obj = torch.utils.data.DataLoader(data_obj, batch_size=opt.batchSize,
                        shuffle=True, num_workers=4)
loader_noobj = torch.utils.data.DataLoader(data_noobj, batch_size=opt.batchSize,
                        shuffle=True, num_workers=4)                        
loader_src = torch.utils.data.DataLoader(data_src, batch_size=opt.batchSize,
                        shuffle=True, num_workers=4)


net = MultiCategoryVAE()
net.load_models(opt.outf, opt.method)
for epoch in range(opt.niter):
    i = 0
    for data_src, data_obj, data_noobj in zip(loader_src, loader_obj, loader_noobj):

        if len(data_src) != opt.batchSize or len(data_obj) != opt.batchSize or len(data_noobj) != opt.batchSize:
            continue

        errD_src, errD_obj, errD_noobj, errG_obj, errG_noobj, err_autoed = net.update_net(data_src, data_obj, data_noobj, i, opt.outf, opt.method)
        print('[%d/%d][%d/%d] Loss: %.4f %.4f %.4f | %.4f %.4f %.4f' % (epoch, opt.niter, i, len(loader_src), errD_src.item(), errD_obj.item(), errD_noobj.item(), errG_obj.item(), errG_noobj.item(), err_autoed.item()))            

        i += 1 
                                                   


# python run_domain_adapt.py --method vae --outf vae_combine --train_model domian_adapt  
# python run_domain_adapt_multi.py --method vae --outf vae_joint                                               