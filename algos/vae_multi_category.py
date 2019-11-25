from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from algos.network_structure import FCDecoder, CnnMultiCategoryDiscriminator
from algos.vae_binary import BinaryVAE
import algos.tool_func as tool

class MultiCategoryVAE(BinaryVAE):
    def __init__(self, obj_encoder_type = 'CNNTarget', latent_length=3, last_conv_channel=16, discriminator_conv_size=16, lr=0.0002 ):
        self.discriminator_conv_size = discriminator_conv_size

        super(MultiCategoryVAE, self).__init__(obj_encoder_type=obj_encoder_type, latent_length=latent_length, last_conv_channel=last_conv_channel, lr=lr)
        self.src_label = 1
        self.obj_label = 0
        self.noobj_label = 2
        self.label_src, self.label_obj, self.label_noobj = None, None, None

    def init_discriminator(self):
        from algos.network_structure import CnnMultiCategoryDiscriminator
        discriminator = CnnMultiCategoryDiscriminator(input_channel=self.last_conv_channel, last_conv_size=self.discriminator_conv_size).to(self.device)
        return discriminator

    def init_discriminator_opt_loss(self, net_discriminator):
        discriminator_params = list(net_discriminator.parameters())
        optimizer_d = torch.optim.Adam(discriminator_params, lr=self.lr)
        criterion_adv = nn.CrossEntropyLoss()
        return optimizer_d, criterion_adv

    def get_discriminator_loss(self, src_img, obj_img, noobj_img):         
        ###########################
        # train to recognize src
        ###########################
        errD_src = self.get_loss_on_label(src_img, self.label_src, self.net_encoder_sourse, self.net_discriminator)
    
        ###########################
        # train to recognize obj
        ###########################  
        errD_obj = self.get_loss_on_label(obj_img, self.label_obj, self.net_encoder_sourse, self.net_discriminator) 

        ###########################
        # train to recognize noobj
        ###########################  
        errD_noobj = self.get_loss_on_label(noobj_img, self.label_noobj, self.net_encoder_sourse, self.net_discriminator)

        errD = errD_src + errD_obj + errD_noobj
        return errD_src, errD_obj, errD_noobj, errD

    def load_models(self, outf, method):
        tool.load_models(outf, method, src_encoder=self.net_encoder_sourse, decoder=self.net_decoder, discriminator=self.net_discriminator)
        # tool.load_models(outf, method, src_encoder=self.net_encoder_sourse, decoder=self.net_decoder)

    def update_discriminator(self, errD_src, errD_obj, errD_noobj):
        errD_src_th, errD_obj_th, errD_noobj_th = 0.8, 0.8, 0.8

        # err_use = errD_src
        # if err_use < errD_obj:
        #     err_use = errD_obj
        # if err_use < errD_noobj:
        #     err_use = errD_noobj

        # if err_use > 0.7:
        #     err_use.backward()
        #     print('update')
        #     self.optimizer_d.step() 
        if errD_src > errD_src_th:
            errD_src.backward()
        if errD_obj > errD_obj_th:
            errD_obj.backward()
        if errD_noobj > errD_noobj_th:
            errD_noobj.backward()
    
        if errD_src > errD_src_th or errD_obj > errD_obj_th or errD_noobj > errD_noobj_th:
            print('update')
            self.optimizer_d.step()        

    def update_generator(self, errG_obj, errG_noobj_1, errG_noobj_2, src_autoed_loss):
        # errG_obj_th, errG_noobj_th = 0.7, 0.7
        # if errG_obj > errG_obj_th:
        #     errG_obj * 0.001
        #     errG_obj.backward()
        # if errG_noobj > errG_noobj_th:
        #     errG_noobj * 0.001
        #     errG_noobj.backward()

        # err = errG_obj + errG_noobj
        # err.backward()
        # self.optimizer_et.step()

        # src_autoed_loss.backward()

        autoed_loss = errG_obj*1 - errG_noobj_1*0 - errG_noobj_2*0 + src_autoed_loss * 0
        autoed_loss.backward()
        self.optimizer_autoed.step()

    def update_net(self, data_src, data_obj, data_noobj, epoch, iteration, outf, method):
        self.net_discriminator.zero_grad()
        self.net_encoder_sourse.zero_grad()
        # self.net_encoder_target.zero_grad()
        self.net_decoder.zero_grad()

        src_img = data_src.to(self.device)
        obj_img = data_obj.to(self.device)
        noobj_img = data_noobj.to(self.device)

        batch_size = src_img.size(0)
        if self.label_src is None:
            self.label_obj = torch.full((batch_size,), self.obj_label, dtype=torch.long, device=self.device)
            self.label_src = torch.full((batch_size,), self.src_label, dtype=torch.long, device=self.device)
            self.label_noobj = torch.full((batch_size,), self.noobj_label, dtype=torch.long, device=self.device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        errD_src, errD_obj, errD_noobj, errD  = self.get_discriminator_loss(src_img, obj_img, noobj_img)
        self.update_discriminator(errD_src, errD_obj, errD_noobj)
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        errG_obj = self.get_loss_on_label(obj_img, self.label_src, self.net_encoder_sourse, self.net_discriminator)
        errG_noobj_1 = self.get_loss_on_label(noobj_img, self.label_obj, self.net_encoder_sourse, self.net_discriminator)
        errG_noobj_2 = self.get_loss_on_label(obj_img, self.label_noobj, self.net_encoder_sourse, self.net_discriminator)

        ############################
        # (3) Update Src AutoED network: 
        ###########################
        src_autoed_loss = self.get_autoed_loss(src_img)
        self.update_generator(errG_obj, errG_noobj_1, errG_noobj_2, src_autoed_loss)

        print('[%d][%d] Loss: %.4f %.4f %.4f | %.4f %.4f %.4f %.4f' % 
            (epoch, iteration, errD_src.item(), errD_obj.item(), errD_noobj.item(), errG_obj.item(), errG_noobj_1.item(), errG_noobj_2.item(), src_autoed_loss.item()))            

        if iteration == 0:
            src_mu, src_conv = self.net_encoder_sourse.get_mu_conv(src_img)
            src_reconst = self.net_decoder(src_mu)

            obj_mu, obj_conv = self.net_encoder_sourse.get_mu_conv(obj_img)
            obj_reconst = self.net_decoder(obj_mu)

            noobj_mu, noobj_conv = self.net_encoder_sourse.get_mu_conv(noobj_img)
            noobj_reconst = self.net_decoder(noobj_mu)

            size = src_conv[0].shape[-1]
            tool.save_result_imgs(outf, method, 
                            src_img = src_img, src_feature = None, src_reconst = src_reconst, src_conv = src_conv[0].view(self.last_conv_channel, 1, size, size),
                            obj_img = obj_img, obj_feature = None, obj_reconst = obj_reconst, obj_conv = obj_conv[0].view(self.last_conv_channel, 1, size, size),
                            noobj_img = noobj_img, noobj_feature = None, noobj_reconst = noobj_reconst, noobj_conv = noobj_conv[0].view(self.last_conv_channel, 1, size, size))
        tool.save_models(outf, method, src_encoder = self.net_encoder_sourse, obj_encoder = None, decoder = self.net_decoder, 
                                            discriminator = self.net_discriminator)
    
        # return errD_src, errD_obj, errD_obj, errG_obj, errG_noobj, src_autoed_loss 