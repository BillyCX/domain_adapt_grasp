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

if opt.method == 'vae':
    # from domain_adapt.net_structure_vae import EncoderSourse, EncoderTarget, Decoder, Discriminator
    import domain_adapt.net_structure_vae as net
elif opt.method == 'spatial':
    from domain_adapt.net_structure_spatial import EncoderSourse, EncoderTarget, Decoder, Discriminator, DecoderTarget
elif opt.method == 'cnn':
    from domain_adapt.net_structure import EncoderSourse, EncoderTarget, Decoder, Discriminator


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

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


data_obj = training_data.get_object_dataset()
data_src = training_data.get_sourse_dataset()

loader_obj = torch.utils.data.DataLoader(data_obj, batch_size=opt.batchSize,
                        shuffle=True, num_workers=4)
loader_src = torch.utils.data.DataLoader(data_src, batch_size=opt.batchSize,
                        shuffle=True, num_workers=4)
                        
torch.autograd.set_detect_anomaly(True)
# tsne_data_src = []
# for i in range(50):
#     index = np.random.randint(len(data_src))
#     sample = data_src[index]
#     tsne_data_src.append(sample.numpy())
# tsne_data_src = np.asarray(tsne_data_src)
# tsne_data_src = torch.from_numpy(tsne_data_src)

# tsne_data_obj = []
# for i in range(100):
#     index = np.random.randint(len(data_obj))
#     sample = data_obj[index]
#     tsne_data_obj.append(sample.numpy())

# tsne_data_src, tsne_data_obj = np.asarray(tsne_data_src), np.asarray(tsne_data_obj)
# tsne_data_src, tsne_data_obj = torch.from_numpy(tsne_data_src), torch.from_numpy(tsne_data_obj)   

# print(tsne_data_src.shape, tsne_data_obj.shape)


# tsne_data_src = torch.utils.data.DataLoader(data_src, batch_size=100, shuffle=True, num_workers=4)
# tsne_data_src = next(iter(tsne_data_src)).detach()
# tsne_data_obj = torch.utils.data.DataLoader(data_obj, batch_size=100, shuffle=True, num_workers=4)
# tsne_data_obj = next(iter(tsne_data_obj)).detach()

net.load_models(opt.outf, opt.method)
net.net_encoder_sourse.eval()
net.net_decoder.eval()

# data = next(iter(loader_src))
# img = data.to('cuda')
# z_tensor, _, _, _ = net.net_encoder_sourse(img)

# z = np.random.normal(0.0, 1.0, size=(64, 32))
# # for i in range(64):
# #     z[i] = [0.6307,  0.4972, -1.6758, -1.1992,  1.0723,  0.0110, -1.1101, -2.1405,
# #          -1.3967,  1.9753,  0.7849, -1.2668, -1.7587,  0.7154,  0.5063,  1.6014,
# #          -0.4185, -0.1919,  1.0143,  0.0240,  0.1138, -0.0537,  2.1328,  1.2988,
# #          -0.2942, -0.0185,  0.5290,  0.0383, -0.4651,  0.6130,  0.0573,  0.6485]
# for i in range(64):
#     z[i][1] = -5 + 10/64.0 * i 

# z_tensor = torch.from_numpy(z)
# print(z_tensor.shape)
# output = net.net_decoder(z_tensor.float().cuda())
# print(z_tensor)
# vutils.save_image(output.detach(),
#                         'results/%s/%s/beta_vae.png' % (opt.outf, opt.method),
#                         normalize=True)

if opt.train_model == 'autoed':
    for epoch in range(opt.niter):
        for i, data in enumerate(loader_src, 0):
            if len(data) != opt.batchSize:
                continue            

            err = net.update_src_autoed(data, i, opt.outf, opt.method)
            print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, opt.niter, i, len(loader_src), err.item()))

            # if i == 0:
            #     feature_src, src_conv = net.get_feature_src(tsne_data_src)
            #     X_2d = tsne.fit_transform(feature_src.cpu().detach().numpy())

            #     plt.clf()
            #     plt.scatter(X_2d[:, 0], X_2d[:, 1], c='r', label='src')
            #     plt.savefig('results/%s/%s/tsne.png' % (opt.outf, opt.method))
         

if opt.train_model == 'domian_adapt':
    # net_encoder_sourse.load_state_dict(torch.load('results/%s/%s/src_auto_g.pth' % (opt.outf, opt.method)))
    # net_decoder.load_state_dict(torch.load('results/%s/%s/src_auto_d.pth' % (opt.outf, opt.method)))
    # net.net_encoder_target.load_state_dict(torch.load('results/%s/%s/net_te.pth' % (opt.outf, opt.method)))
    # net.net_discriminator.load_state_dict(torch.load('results/%s/%s/net_dis.pth' % (opt.outf, opt.method)))

    for epoch in range(opt.niter):
        i = 0
        for data_src, data_obj in zip(loader_src, loader_obj):
            # net.net_encoder_target.train()
            # net.net_discriminator.train()

            if len(data_src) != opt.batchSize or len(data_obj) != opt.batchSize:
                continue

            err = net.update_src_autoed(data_src, i, opt.outf, opt.method)
            print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, opt.niter, i, len(loader_src), err.item()))

            errD, errG = net.update_domain_adapt(data_src, data_obj, i, opt.outf, opt.method)
            print('[%d/%d][%d/%d] Loss: %.4f %.4f ' % (epoch, opt.niter, i, len(loader_src), errD.item(), errG.item()))

            # if i == 0:
            #     net.net_encoder_target.eval()
            #     net.net_discriminator.eval()       

            #     # feature_src, src_conv = net.get_feature_src(tsne_data_src)
            #     # src_f = src_conv.view(100, 16*8*8).cpu().detach().numpy()
            #     src_f = tsne_data_src.view(100, 3*64*64).cpu().detach().numpy()
            #     X_2d_src = tsne.fit_transform(src_f)

            #     # obj_conv = net.get_feature_obj(tsne_data_obj)
            #     # obj_f = obj_conv.view(100, 16*8*8).cpu().detach().numpy()
            #     obj_f = tsne_data_obj.view(100, 3*64*64).cpu().detach().numpy()
            #     X_2d_obj = tsne.fit_transform(obj_f)

            #     # plt.figure(figsize=(6, 5))
            #     plt.clf()
            #     plt.scatter(X_2d_src[:, 0], X_2d_src[:, 1], c='r', label='src')
            #     plt.scatter(X_2d_obj[:, 0], X_2d_obj[:, 1], c='b', label='obj')

            #     plt.savefig('results/%s/%s/tsne.png' % (opt.outf, opt.method))
            #     # plt.legend()
            #     # plt.show()
            #     # input()                

            i += 1 
                                                   


# python run_domain_adapt.py --method vae --outf vae_combine --train_model domian_adapt                                                   