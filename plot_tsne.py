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

from algos.vae_multi_binary import MultiCategoryVAE
from algos.vae_spatial import SpatialVAE
from algos.vae_multi_binary_adda import MultiCategoryADDA

from algos import data_loader as training_data
from matplotlib import pyplot as plt
import algos.tool_func as tool

from sklearn.manifold import TSNE
tsne = TSNE(random_state=25111990, perplexity=50, learning_rate=100, n_iter=5000)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width osf the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
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

training_data.exp_type = 'cup'

data_src = training_data.get_dataset(img_type='src', root_dir='./dataset_cup/src/', load_label=True)
data_obj = training_data.get_dataset(img_type='obj', root_dir='./dataset_cup/obj/')
data_noobj = training_data.get_dataset(img_type='noobj', root_dir='./dataset/noobj/')

# loader_obj_only = torch.utils.data.DataLoader(data_obj_only, batch_size=opt.batchSize,
#                         shuffle=True, num_workers=4)
batch_size = 300
loader_src = torch.utils.data.DataLoader(data_src, batch_size=batch_size,
                        shuffle=True, num_workers=16)
loader_obj = torch.utils.data.DataLoader(data_obj, batch_size=batch_size,
                        shuffle=True, num_workers=16)
loader_noobj = torch.utils.data.DataLoader(data_noobj, batch_size=batch_size,
                        shuffle=True, num_workers=16)                         


device = torch.device("cpu")
method = 'adda'
if method == 'spatial':
    net = SpatialVAE(device='cpu')
elif method == 'adda':
    net = MultiCategoryADDA(device='cpu')
else:
    net = MultiCategoryVAE(device='cpu')
folder_name='cup_adda_ori'
net.load_models('cup_test', 'adda_ori')
net.net_encoder_target.eval()
net.net_decoder.eval() 

data_src = next(iter(loader_src))
data_obj = next(iter(loader_obj))
data_noobj = next(iter(loader_noobj))

src_img = data_src['image'].to(device)
obj_img = data_obj['image'].to(device)
noobj_img = data_noobj['image'].to(device)

src_img = torch.min(src_img, noobj_img)

# net.net_encoder_target.cpu()
# net.net_decoder.cpu() 
net.net_encoder_target.eval()
net.net_decoder.eval() 

src_z, src_conv = net.net_encoder_target(src_img)
obj_z, obj_conv = net.net_encoder_target(obj_img)
noobj_z, noobj_conv = net.net_encoder_target(noobj_img)

if method == 'spatial':
    src_z, src_feature = net.net_encoder_sourse.get_softmax_feature(src_img, src_conv)
    obj_z, obj_feature = net.net_encoder_sourse.get_softmax_feature(obj_img, obj_conv)
    noobj_z, noobj_feature = net.net_encoder_sourse.get_softmax_feature(noobj_img, noobj_conv)
    size = src_conv[0].shape
    tool.save_test_imgs('result_plot', folder_name, src_img, src_conv[0].view(-1, 1, size[1], size[2]), src_feature.view(-1, 3, 60, 105), ext='src')
    tool.save_test_imgs('result_plot', folder_name, src_img, obj_conv[0].view(-1, 1, size[1], size[2]), obj_feature.view(-1, 3, 60, 105), ext='obj')
    tool.save_test_imgs('result_plot', folder_name, src_img, noobj_conv[0].view(-1, 1, size[1], size[2]), noobj_feature.view(-1, 3, 60, 105), ext='noobj')
else:
    size = src_conv[0].shape
    # src_z = src_z * 0.6 + noobj_z * 0.4
    tool.save_test_imgs('result_plot', folder_name, src_img, src_z.view(-1, 1, size[1], size[2]), src_conv, ext='src')
    tool.save_test_imgs('result_plot', folder_name, obj_img, obj_z.view(-1, 1, size[1], size[2]), obj_conv, ext='obj')
    tool.save_test_imgs('result_plot', folder_name, noobj_img, noobj_z.view(-1, 1, size[1], size[2]), noobj_conv, ext='noobj')

data = np.concatenate([src_z.detach().numpy().reshape(batch_size, -1), obj_z.detach().numpy().reshape(batch_size, -1), noobj_z.detach().numpy().reshape(batch_size, -1)])
print(data.shape)

X_2d = tsne.fit_transform(data)

vutils.save_image(src_img,'results/result_plot/'+ folder_name + '/srctest_img.png', normalize=True) 
vutils.save_image(obj_img,'results/result_plot/'+ folder_name + '/objtest_img.png', normalize=True) 
vutils.save_image(noobj_img,'results/result_plot/'+ folder_name + '/noobjtest_img.png', normalize=True) 
np.save('results/result_plot/'+ folder_name + '/tsne_data.npy',X_2d)

plt.clf()
plt.scatter(X_2d[0:batch_size, 0], X_2d[0:batch_size, 1], c='r', label='src', s=5)
plt.scatter(X_2d[batch_size:batch_size*2, 0], X_2d[batch_size:batch_size*2, 1], c='g', label='obj', s=5)
plt.scatter(X_2d[batch_size*2:batch_size*3, 0], X_2d[batch_size*2:batch_size*3, 1], c='b', label='noobj', s=5)

plt.savefig('results/result_plot/'+ folder_name + '/tsne_plot.png')
# plt.legend()
plt.show()
input()   