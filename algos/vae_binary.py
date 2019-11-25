from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import algos.tool_func as tool

class BinaryVAE():
    def __init__(self, obj_encoder_type = 'CNNLatentDistrib', latent_length=3, last_conv_channel=16, lr=0.0002, device="cuda:0" ):
        self.src_label = 1
        self.obj_label = 0
        self.latent_length = latent_length
        self.last_conv_channel = last_conv_channel
        self.lr = lr
        self.label_obj, self.label_src = None, None
        self.count = 0

        # create networks
        self.device = torch.device(device) #torch.device("cuda:0" if opt.cuda else "cpu")                                    EdgeCrop(130, 70, 30, 60),

        self.net_encoder_sourse = self.init_encoder(device=device)
        print(self.net_encoder_sourse)
        self.net_encoder_target = self.init_encoder()
        print(self.net_encoder_target)
        self.net_decoder = self.init_decoder()
        print(self.net_decoder)
        # net_decoder_target = DecoderTarget().to(device)
        # print(net_decoder_target)
        self.net_discriminator = self.init_discriminator()
        print(self.net_discriminator)

        # self.net_sourse_mask = self.init_mask()
        # print(self.net_sourse_mask)

        # autoencoder loss
        autoencoder_params = list(self.net_encoder_sourse.parameters()) + list(self.net_decoder.parameters()) #+ list(self.net_sourse_mask.parameters())
        # autoencoder_params = list(self.net_decoder.parameters())
        self.optimizer_autoed = torch.optim.Adam(autoencoder_params, lr=lr*0.5)
        self.criterion_autoed = nn.MSELoss()

        self.optimizer_d, self.criterion_adv = self.init_discriminator_opt_loss(self.net_discriminator)

        # # mask loss 
        # mask_params = list(self.net_sourse_mask.parameters())
        # self.optimizer_mask = torch.optim.Adam(mask_params, lr=self.lr)

        # target encoder loss
        # targetencoder_params = list(net_encoder_target.parameters()) + list(net_decoder_target.parameters())
        targetencoder_params = list(self.net_encoder_target.parameters())
        self.optimizer_et = torch.optim.Adam(targetencoder_params, lr=lr)

        # self.net_encoder_target = self.net_encoder_sourse
        # self.optimizer_et = self.optimizer_autoed

        # discriminator loss
        # discriminator_params = list(self.net_discriminator.parameters())
        # self.optimizer_d = torch.optim.Adam(discriminator_params, lr=lr)
        # self.criterion_adv = nn.BCELoss()

    def init_encoder(self, type='CNNLatentDistrib', device="cuda:0"):
        if type == 'CNNLatentDistrib':
            from algos.network_structure import CNNLatentDistrib
            return CNNLatentDistrib(latent_n=self.latent_length, last_conv_channel=self.last_conv_channel).to(self.device)
        elif type == 'CNNTarget':
            from algos.network_structure import CNNTarget
            return CNNTarget(latent_n=self.latent_length, last_conv_channel=self.last_conv_channel).to(self.device)

    def init_decoder(self):
        from algos.network_structure import FCDecoder, CNNSpatial, FCTrajectoryDecoder, CNNTrajectoryDecoder
        return FCTrajectoryDecoder(input_length=self.latent_length).to(self.device)
        # return CNNTrajectoryDecoder(input_channel=self.last_conv_channel).to(self.device)

    def init_discriminator(self):
        from algos.network_structure import CnnBinaryDiscriminatorB as discriminator
        discriminator = discriminator(input_channel=self.last_conv_channel).to(self.device)
        return discriminator

    def init_discriminator_opt_loss(self, net_discriminator):
        discriminator_params = list(net_discriminator.parameters())
        optimizer_d = torch.optim.Adam(discriminator_params, lr=self.lr)
        criterion_adv = nn.BCELoss()
        return optimizer_d, criterion_adv

    def load_models(self, outf, method):
        tool.load_models(outf, method, src_encoder=self.net_encoder_sourse, decoder=self.net_decoder, discriminator=None)
    
    def decode_obj(self, obj_conv):
        # obj_mu = net_encoder_sourse.fc1(obj_conv.view(-1, 16*8*8))
        size = obj_conv[0].shape[-1]
        obj_z, obj_mu, _ = self.net_encoder_target.bottleneck(obj_conv.view(-1, self.last_conv_channel*size*size))  
        # print(obj_mu.shape) 
        obj_decoded = self.net_decoder(obj_mu)
        return obj_decoded

    def loss_function(self, recon_x, x, mu, logvar, beta):
        BCE = self.criterion_autoed(recon_x, x)
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        KLD = torch.mean(KLD_batch)
    
        # print(BCE.item(), KLD.item())
        # print(torch.mean(logvar.exp(), dim=0), torch.mean(mu, dim=0))
        return BCE, KLD, BCE + KLD*0.0

    def get_autoed_loss(self, encoder, src_img):
        encoded_f, conv = encoder(src_img)
        mu, logvar = encoder.get_mu_std(src_img)
        # print(encoded_f.shape)
        # print('mu', encoded_f[0], mu[0])
        # print('mu', encoded_f[0])
        output = self.net_decoder(encoded_f)
        rec_err, kld, total_err = self.loss_function(output, src_img, mu, logvar, 0) #criterion_autoed(output, img)

        # errG = self.get_loss_on_label(src_img, self.label_obj, self.net_encoder_sourse, self.net_discriminator)
        return total_err

    def get_loss_on_label(self, conv, label, discriminator, detach=False):
        # f, conv = encoder(img)
        if detach:
            discriminator_result = discriminator(conv.detach())
        else:
            discriminator_result = discriminator(conv)
        err = self.criterion_adv(discriminator_result, label)
        return err

    def get_discriminator_loss(self, src_img, obj_img, net_discriminator):  
        errD_src, _ = self.get_loss_on_label(src_img, self.label_src, self.net_encoder_sourse, net_discriminator, detach=True)
        errD_obj, _ = self.get_loss_on_label(obj_img, self.label_obj, self.net_encoder_target, net_discriminator, detach=True) 
        errD = errD_src + errD_obj
        return errD          

    def update_net(self, data_src, data_obj, epoch, iteration, outf, method):
        self.count += 1
        self.net_discriminator.zero_grad()
        self.net_encoder_sourse.zero_grad()
        # self.net_encoder_target.zero_grad()
        self.net_decoder.zero_grad()

        src_img = data_src['image'].to(self.device)
        obj_img = data_obj['image'].to(self.device)

        batch_size = src_img.size(0)
        if self.label_obj is None:
            self.label_obj = torch.full((batch_size,), self.obj_label, device=self.device)
            self.label_src = torch.full((batch_size,), self.src_label, device=self.device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        errD = self.get_discriminator_loss(src_img, obj_img, self.net_discriminator)
        errD.backward()
   
        if errD > 1:
            self.optimizer_d.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        errG, _ = self.get_loss_on_label(obj_img, self.label_src, self.net_encoder_sourse, self.net_discriminator)

        # errG.backward()
        # self.optimizer_et.step()

        ############################
        # (3) Update Src AutoED network: 
        ###########################
        src_autoed_loss = self.get_autoed_loss(self.net_encoder_sourse, src_img)
        autoed_loss = errG * 0.001 + src_autoed_loss * 1
        autoed_loss.backward()
        self.optimizer_autoed.step()

        if iteration == 0:
            src_mu, src_conv = self.net_encoder_sourse.get_mu_conv(src_img)
            src_reconst = self.net_decoder(src_mu)

            obj_mu, obj_conv = self.net_encoder_sourse.get_mu_conv(obj_img)
            obj_reconst = self.decode_obj(obj_mu) #self.net_decoder(obj_mu)

            size = src_conv[0].shape[-1]
            tool.save_result_imgs(outf, method, 
                            src_img = src_img, src_feature = None, src_reconst = src_reconst, src_conv = src_conv[0].view(self.last_conv_channel, 1, size, size),
                            obj_img = obj_img, obj_feature = None, obj_reconst = obj_reconst, obj_conv = obj_conv[0].view(self.last_conv_channel, 1, size, size))
                            
            tool.save_models(outf, method, src_encoder = self.net_encoder_sourse, obj_encoder = None, decoder = self.net_decoder, 
                                            discriminator = self.net_discriminator)
    
        return errD, errG, src_autoed_loss 