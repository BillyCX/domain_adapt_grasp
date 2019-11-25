from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn

from matplotlib import pyplot as plt
from algos.data_loader import ExpDataProcessor
from algos.vae_multi_binary import MultiCategoryVAE
from algos.vae_multi_binary_adda import MultiCategoryADDA

from algos.vae_spatial import SpatialVAE

import torchvision.utils as vutils

import sys, time
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import matplotlib.pyplot as plt
from skimage import io, transform

from PIL import Image
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=int, help='number of data loading workers', default=1)
opt = parser.parse_args()
exp_id = opt.exp_id
print(exp_id)

manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

model_device = torch.device("cuda:0")
# # Create and set logger
# logger = createConsoleLogger(LoggerLevel.Debug)
# setGlobalLogger(logger)

# fn = Freenect2()
# num_devices = fn.enumerateDevices()
# if num_devices == 0:
#     print("No device connected!")
#     sys.exit(1)

# serial = fn.getDeviceSerialNumber(0)
# device = fn.openDevice(serial, pipeline=pipeline)

# listener = SyncMultiFrameListener(FrameType.Color)

# # Register listeners
# device.setColorFrameListener(listener)
# device.setIrAndDepthFrameListener(listener)

# device.start()

# # NOTE: must be called after device.start()
# registration = Registration(device.getIrCameraParams(),
#                             device.getColorCameraParams())

img_processor = ExpDataProcessor()

# exp_id = 1
load_path = 'cup_test'

if exp_id == 1:
    method = 'vae'
elif exp_id == 2 or exp_id == 3:
    method = 'adda'
else:
    method = 'spatial'

if method == 'spatial':
    net = SpatialVAE()
elif method == 'adda':
    net = MultiCategoryADDA()
else:
    net = MultiCategoryVAE()


if exp_id == 1:
    net.load_models(load_path, 'ours')
elif exp_id == 2:
    net.load_models(load_path, 'adda_ori')
elif exp_id == 3:
    net.load_models(load_path, 'adda_extend')
elif exp_id == 4:
    net.load_models(load_path, 'gplac_ori')
else:
    net.load_models(load_path, 'gplac_extend')

# net.load_models('pen_test', 'adda_extend')
net.net_encoder_sourse.eval()
net.net_encoder_target.eval()
net.net_decoder.eval() 

# while True:
#     frames = listener.waitForNewFrame()

#     color = frames["color"].asarray()[:,:,:3]
#     color = color[...,::-1]

#     # color = transform.resize(color, (540, 960))
#     # color = np.fliplr(color)

#     listener.release(frames)
#     break

# device.stop()
# device.close()

color = io.imread('/home/xi/exp_adda_ori/1569078494.8282123src.png')

data_test = img_processor.process(color)
print(data_test['image'].shape)

# img_in = data_test['image'].numpy()
# plt.clf()
# plt.imshow(img_in) #Needs to be in row,col order
# # plt.pause(0.1)
# plt.show()

test_img = data_test['image'].to(model_device).view(1, 3, 60, 105)
vutils.save_image(test_img,'/home/xi/exp_adda_ori/'+str(time.time())+'src.png', normalize=True)

if method == 'spatial':
    _, test_conv = net.net_encoder_target(test_img.detach())
    key, feature = net.net_encoder_target.get_softmax_feature(test_img, test_conv.detach())
    pred_label = net.net_decoder(key.detach()).detach().cpu().numpy()
    pred_label = img_processor.unnormalize_result(pred_label)[0]
    print(pred_label, test_conv.shape)
    vutils.save_image(feature,'/home/xi/exp_adda_ori/'+str(time.time())+'.png', normalize=True)
else:
    # test_img = data_test['image'].to(model_device).view(1, 3, 60, 105)
    test_z, test_conv = net.net_encoder_target(test_img.detach())
    pred_label = net.net_decoder(test_z.detach()).detach().cpu().numpy()
    pred_label = img_processor.unnormalize_result(pred_label)[0]
    print(pred_label, test_conv.shape)
    vutils.save_image(test_conv,'/home/xi/exp_adda_ori/'+str(time.time())+'.png', normalize=True)    

# size = src_conv[0].shape

f= open("/home/xi/test.txt","w")
f.write(str(pred_label[0]) + ' ' + str(pred_label[1]) + ' ' + str(pred_label[2]))

# sys.exit(0)


