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

from algos import data_loader as training_data

from matplotlib import pyplot as plt
import algos.tool_func as tool

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)

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
device = torch.device("cuda:0")

# loader_obj = []
# obj_index = [1, 2, 3, 4, 12, 13, 14, 15]
# obj_num = len(obj_index)
# obj_weight = np.ones(obj_num) * 10

# for i in obj_index:
#     obj_path = './dataset/obj/' + 'obj' + str(i) + '/'
#     data_obj = training_data.get_dataset(root_dir=obj_path)
#     loader = torch.utils.data.DataLoader(data_obj, batch_size=opt.batchSize,
#                             shuffle=True, num_workers=16)
#     loader_obj.append(loader)

# data_obj = training_data.get_dataset(root_dir='./dataset/obj/')
# loader = torch.utils.data.DataLoader(data_obj, batch_size=opt.batchSize,
#                         shuffle=True, num_workers=16)
# loader_obj.append(loader)

# data_obj_only = training_data.get_object_dataset(root_dir='./dataset/obj/obj')
data_src = training_data.get_dataset(img_type='src', root_dir='./dataset_cup/src/', load_label=True)
data_obj = training_data.get_dataset(img_type='obj', root_dir='./dataset_cup/obj/')
data_noobj = training_data.get_dataset(img_type='noobj', root_dir='./dataset/noobj/')
data_test = training_data.get_dataset(img_type='test', root_dir='./dataset_cup/test/', load_label=True)

# loader_obj_only = torch.utils.data.DataLoader(data_obj_only, batch_size=opt.batchSize,
#                         shuffle=True, num_workers=4)
loader_src = torch.utils.data.DataLoader(data_src, batch_size=opt.batchSize,
                        shuffle=True, num_workers=16)
loader_obj = torch.utils.data.DataLoader(data_obj, batch_size=opt.batchSize,
                        shuffle=True, num_workers=16)
loader_noobj = torch.utils.data.DataLoader(data_noobj, batch_size=opt.batchSize,
                        shuffle=True, num_workers=16)                        
loader_test = torch.utils.data.DataLoader(data_test, batch_size=100,
                        shuffle=True, num_workers=16)    


# for data_src in loader_noobj:
#     print('1')
#     break

# for data_src in loader_src: #zip(loader_obj, loader_noobj):
#     print(len(data_src['image']))
#     print('--------------------dd') 

# if opt.method == 'vae_binary':
#     from algos.domain_adapt_vae_binary import BinaryVAE
#     net = BinaryVAE()

# if opt.method == 'vae_multi':
#     from algos.vae_multi_category import MultiCategoryVAE
#     net = MultiCategoryVAE()
if opt.method == 'vae_multi':
# elif opt.method == 'vae_multi_binar':
    from algos.vae_multi_binary import MultiCategoryVAE
    net = MultiCategoryVAE()

elif opt.method == 'spatial':
    from algos.vae_spatial import SpatialVAE
    net = SpatialVAE()
    print('spatial encoder')

elif opt.method == 'adda':
    from algos.vae_multi_binary_adda import MultiCategoryADDA
    net = MultiCategoryADDA()
    print('spatial encoder')

batch_loss = []
batch_loss_xyt = []
batch_loss_xyt_std = []
model_loss = []
model_batch_loss = []

net.load_models(opt.outf, opt.method)
batch_loss = np.load('results/%s/%s/test_loss.npy' % (opt.outf, opt.method)).tolist()
batch_loss_xyt = np.load('results/%s/%s/test_loss_xyt.npy' % (opt.outf, opt.method)).tolist()
batch_loss_xyt_std = np.load('results/%s/%s/test_loss_xyt_std.npy' % (opt.outf, opt.method)).tolist()
model_loss = np.load('results/%s/%s/model_loss.npy' % (opt.outf, opt.method)).tolist()

min_loss = 100

for epoch in range(500):
    i = 0
    data_test = next(iter(loader_test))

    # net.net_encoder_target.cpu()
    # net.net_decoder.cpu() 
    net.net_encoder_target.eval()
    net.net_decoder.eval() 

    test_img = data_test['image'].to(device)
    test_label = data_test['label'].to(device)

    # test_z, test_conv = net.net_encoder_target(test_img.detach())
    test_z, test_conv = net.net_encoder_target(test_img.detach())
    # test_mu, test_mu_conv = net.net_encoder_target.get_mu_conv(test_img.detach())

    if opt.method == 'spatial':
        key, src_feature = net.net_encoder_sourse.get_softmax_feature(test_img, test_conv)
        pred_label = net.net_decoder(key.detach()).detach()
        size = test_conv[0].shape
        tool.save_test_imgs(opt.outf, opt.method, test_img, test_conv[0].view(-1, 1, size[1], size[2]), src_feature.view(-1, 3, 60, 105), ext='')
        test_loss = net.get_autoed_loss(net.net_encoder_target, test_conv.detach(), test_label.detach()).detach().item()
    else:
        pred_label = net.net_decoder(test_z.detach()).detach()
        size = test_conv[0].shape
        tool.save_test_imgs(opt.outf, opt.method, test_img, test_z.view(-1, 1, size[1], size[2]), test_conv, ext='')
        test_loss = net.get_autoed_loss(net.net_encoder_target, test_z.detach(), test_label.detach()).detach().item()

    diff = (pred_label - test_label).detach().cpu().numpy()
    xyt_mse = (np.abs(diff)).mean(axis=0)
    xyt_std = diff.std(axis=0)

    print(pred_label[0], test_label[0], diff[0], np.mean(np.square(diff)[:,2]))

    batch_loss.append(test_loss)
    batch_loss_xyt.append(xyt_mse)
    batch_loss_xyt_std.append(xyt_std)
    model_loss.append(np.mean(model_batch_loss))

    # net.net_encoder_target.to(device)
    # net.net_decoder.to(device)
    net.net_encoder_target.train()
    net.net_decoder.train() 


    plt.clf()
    plt.plot(np.array(batch_loss), color='b')
    plt.plot(np.array(model_loss), color='r')
    plt.gca().set_ylim([0, 0.15])
    plt.yticks(np.arange(0, 0.15, 0.05))
    plt.savefig('results/%s/%s/test_loss.png' % (opt.outf, opt.method))
    np.save('results/%s/%s/test_loss' % (opt.outf, opt.method), batch_loss)
    np.save('results/%s/%s/model_loss' % (opt.outf, opt.method), model_loss)


    plt.clf()
    plt.plot(np.array(batch_loss_xyt)[:,0], color='r')
    plt.plot(np.array(batch_loss_xyt)[:,1], color='g')
    plt.plot(np.array(batch_loss_xyt)[:,2], color='b')
    plt.gca().set_ylim([0, 0.5])
    plt.yticks(np.arange(0, 0.5, 0.05))    
    plt.savefig('results/%s/%s/test_loss_xyt.png' % (opt.outf, opt.method))
    np.save('results/%s/%s/test_loss_xyt' % (opt.outf, opt.method), batch_loss_xyt)

    plt.clf()
    plt.plot(np.array(batch_loss_xyt_std)[:,0], color='r')
    plt.plot(np.array(batch_loss_xyt_std)[:,1], color='g')
    plt.plot(np.array(batch_loss_xyt_std)[:,2], color='b')
    plt.gca().set_ylim([0, 0.5])
    plt.yticks(np.arange(0, 0.5, 0.05))    
    plt.savefig('results/%s/%s/test_loss_xyt_std.png' % (opt.outf, opt.method))
    np.save('results/%s/%s/test_loss_xyt_std' % (opt.outf, opt.method), batch_loss_xyt_std)

    model_batch_loss = []

    for data_src, data_obj, data_noobj in zip(loader_src, loader_obj, loader_noobj):
    # for data_src, data_obj, data_noobj in zip(loader_src, loader_obj[i%obj_num], loader_noobj):
        # data_obj = next(iter(loader_obj[i%obj_num]))
        # print(i%obj_num)
        if len(data_src['image']) != opt.batchSize or len(data_obj['image']) != opt.batchSize or len(data_noobj['image']) != opt.batchSize:
            continue

        loss = net.update_net(data_src, data_obj, data_noobj, epoch, i, opt.outf, opt.method)
        # print('[%d/%d][%d/%d] Loss: %.4f %.4f %.4f | %.4f %.4f %.4f' % (epoch, opt.niter, i, len(loader_src), errD_src.item(), errD_obj.item(), errD_noobj.item(), errG_obj.item(), errG_noobj.item(), err_autoed.item()))            
        model_batch_loss.append(loss)
        i += 1 
                                    



# python run_domain_adapt.py --method vae --outf vae_combine --train_model domian_adapt  
# python run_domain_adapt_multi.py --method vae --outf vae_joint      
# python run_exp.py --method vae_multi --outf vae_joint