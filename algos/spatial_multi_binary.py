from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from algos.network_structure import FCDecoder, CnnMultiCategoryDiscriminator
from algos.domain_adapt_vae_multi_category import MultiCategoryVAE
import algos.tool_func as tool

class MultiCategorySpatial(MultiCategoryVAE):
    def __init__(self, latent_length=32, last_conv_channel=16, discriminator_conv_size=53, lr=0.0002 ):
        super(MultiCategorySpatial, self).__init__(latent_length=latent_length, last_conv_channel=last_conv_channel, discriminator_conv_size=discriminator_conv_size, lr=lr)

    def init_encoder(self):
        from algos.network_structure import CNNSpatial
        return CNNSpatial().to(self.device)

    def get_autoed_loss(self, src_img):
        encoded_f, conv = self.net_encoder_sourse(src_img)
        output = self.net_decoder(encoded_f)
        BCE = self.criterion_autoed(output, src_img)
        return BCE

    def update_discriminator(self, errD_src, errD_obj, errD_noobj):
        errD_src_th, errD_obj_th, errD_noobj_th = 0.7, 0.7, 0.7
        if errD_src > errD_src_th:
            errD_src.backward()
        if errD_obj > errD_obj_th:
            errD_obj.backward()
        if errD_noobj > errD_noobj_th:
            errD_noobj.backward()
        # errD.backward()
    
        if errD_src > errD_src_th or errD_obj > errD_obj_th or errD_noobj > errD_noobj_th:
            self.optimizer_d.step() 

    def update_generator(self, errG_obj, errG_noobj, src_autoed_loss):
        errG_obj_th, errG_noobj_th = 0.7, 0.7
        if errG_obj > errG_obj_th:
            errG_obj * 0.005
            errG_obj.backward()
        if errG_noobj > errG_noobj_th:
            errG_noobj * 0.005
            errG_noobj.backward()

        src_autoed_loss.backward()
        # autoed_loss = errG_obj*0.0001 + errG_noobj*0.0001 + src_autoed_loss * 1
        # autoed_loss.backward()
        self.optimizer_autoed.step()

