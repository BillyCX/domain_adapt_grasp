from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from algos.network_structure import FCDecoder, CnnBinaryDiscriminator, CnnBinaryDiscriminator_ours, CnnBinaryDiscriminatorB
from algos.vae_binary import BinaryVAE
import algos.tool_func as tool

class MultiCategoryVAE(BinaryVAE):
    def __init__(self, obj_encoder_type = 'CNNLatentDistrib', latent_length=3, last_conv_channel=1, discriminator_conv_size=16, lr=0.0002, device="cuda:0"):
        self.discriminator_conv_size = discriminator_conv_size

        super(MultiCategoryVAE, self).__init__(obj_encoder_type=obj_encoder_type, latent_length=latent_length, last_conv_channel=last_conv_channel, lr=lr, device=device)
        self.src_label = 0.9
        self.obj_label = 0.1
        self.noobj_label = 0.9
        self.label_src, self.label_obj, self.label_noobj = None, None, None
        self.weight = torch.from_numpy(np.array([1, 1, 1])).to(self.device).float()

        self.net_discriminator_noobj = CnnBinaryDiscriminator(input_channel=last_conv_channel).to(device) #self.init_discriminator() self.init_discriminator() #CropCnnBinaryDiscriminator(input_channel=last_conv_channel).to(device) #self.init_discriminator()
        print(self.net_discriminator_noobj)
        self.optimizer_d_noobj, self.criterion_adv_noobj = self.init_discriminator_opt_loss(self.net_discriminator_noobj)

        self.net_encoder_target = self.net_encoder_sourse
        self.optimizer_et = self.optimizer_autoed

        # # autoencoder loss
        # autoencoder_params = list(self.net_encoder_sourse.parameters())
        # self.optimizer_autoed = torch.optim.Adam(autoencoder_params, lr=lr)
        # self.criterion_autoed = nn.MSELoss()

        # self.optimizer_d, self.criterion_adv = self.init_discriminator_opt_loss(self.net_discriminator)

        # self.net_encoder_target = self.net_encoder_target
        # self.optimizer_et = self.optimizer_autoed

    def init_discriminator(self):
        from algos.network_structure import CnnBinaryDiscriminatorB as discriminator
        discriminator = discriminator(input_channel=self.last_conv_channel).to(self.device)
        return discriminator

    def load_models(self, outf, method):
        tool.load_models(outf, method, src_encoder=self.net_encoder_sourse, decoder=self.net_decoder, 
                                    discriminator_noobj=self.net_discriminator_noobj, discriminator=self.net_discriminator)
        # self.net_sourse_mask.load_state_dict(torch.load('results/%s/%s/net_mask.pth' % (outf, method)))
        # tool.load_models(outf, method, obj_encoder=self.net_encoder_target, discriminator_noobj=self.net_discriminator_noobj, discriminator=self.net_discriminator)

        # self.net_encoder_target.load_state_dict(torch.load('results/%s/%s/net_se.pth' % (outf, method)))

    def get_discriminator_loss(self, src_conv, obj_conv, noobj_conv):         
        ###########################
        # train to recognize src
        ###########################
        errD_src = self.get_loss_on_label(src_conv, self.label_src, self.net_discriminator, detach=True)
    
        ###########################
        # train to recognize obj
        ###########################  
        errD_obj = self.get_loss_on_label(obj_conv, self.label_obj, self.net_discriminator, detach=True) 


        ###########################
        # train to recognize obj
        # ########################### 
        errD2_obj = self.get_loss_on_label(obj_conv, self.label_obj, self.net_discriminator_noobj, detach=True) 

        ###########################
        # train to recognize noobj
        ###########################  
        errD2_noobj = self.get_loss_on_label(noobj_conv, self.label_noobj, self.net_discriminator_noobj, detach=True)

        # print(dis_obj)
        # print(dis_obj2)
        # errD = errD_src + errD_obj + errD_noobj
        return errD_src, errD_obj, errD2_obj, errD2_noobj

    def update_discriminator(self, errD_src, errD_obj, errD2_obj, errD2_noobj):
        err1 = errD_src + errD_obj
        err2 = errD2_obj + errD2_noobj
        update_d_noobj = False
        update_d = False

        if err1 > 1.2:
            err1.backward()
            self.optimizer_d.step()
            update_d = True
            print('update d')
        if err2 > 0.65:
            err2.backward()
            self.optimizer_d_noobj.step()      
            update_d_noobj = True  
            print('update d noobj')
        return update_d, update_d_noobj   

    def update_generator(self, errG_obj, errG_src, errG_obj_2, errG_noobj_2, src_autoed_loss):
        # errG_obj_th, errG_noobj_th = 0.7, 0.7
        # if errG_obj > errG_obj_th:
        #     errG_obj * 0.001
        #     errG_obj.backward()
        # if errG_noobj > errG_noobj_th:
        #     errG_noobj * 0.001
        #     errG_noobj.backward()

        # err = errG_obj + errG_obj_2 + errG_noobj_2
        # err.backward()
        # self.optimizer_et.step()

        w = 0.005
        w2 = w*10
        if errG_obj_2 + errG_noobj_2 < 0.65:
            w2 = 0
        print(w2)
        autoed_loss = errG_src*w + errG_obj*w + src_autoed_loss*1 + errG_obj_2*w2 + errG_noobj_2*w2
        # autoed_loss = errG_src * 1 + src_autoed_loss*1
        autoed_loss.backward()
        self.optimizer_autoed.step()

    def get_autoed_loss(self, encoder, src_z, src_label, noobj_f=None):
        # encoded_f, conv = encoder(src_img)
        # conv_masked = (conv + self.mask)/2
        # size = src_conv[0].shape
        # z, mu, _ = encoder.bottleneck(src_conv.view(-1, self.last_conv_channel*size[1]*size[2]))

        # randomize on feature space using obstacle image
        if noobj_f is not None:
            z_used = torch.zeros_like(src_z)
            # encoded_noobj, conv = encoder(noobj_img)

            rand_w = 0.3 + np.random.rand()*0.3
            z_used = src_z * (1-rand_w) + noobj_f.detach()*rand_w
            # for i in range(len(z)):
                # rand_w = 0.1 + np.random.rand()*0.9
                # z_used[i] = z[i] * (1-rand_w) + encoded_noobj[i].detach()*rand_w
        else:
            z_used = src_z

        output = self.net_decoder(z_used)
        # output = self.net_decoder(encoded_f)

        # mu, logvar = encoder.get_mu_std(src_img)
        # rec_err, kld, total_err = self.loss_function(output, src_label, mu, logvar, 0) #criterion_autoed(output, img)
        BCE = self.criterion_autoed(output, src_label)
        # BCE = self.criterion_autoed(output[:,:-1], src_label[:,:-1])

        return BCE

    def set_eval_mode(self):
        self.net_discriminator.eval()
        self.net_discriminator_noobj.eval()
        self.net_encoder_sourse.eval()
        self.net_encoder_target.eval()
        # self.net_sourse_mask.eval()
        self.net_decoder.eval()        

    def set_train_mode(self):
        self.net_discriminator.train()
        self.net_discriminator_noobj.train()
        self.net_encoder_sourse.train()
        self.net_encoder_target.train()
        # self.net_sourse_mask.train()
        self.net_decoder.train()       

    def update_net(self, data_src, data_obj, data_noobj, epoch, iteration, outf, method):
        self.count += 1
        self.net_discriminator.zero_grad()
        self.net_discriminator_noobj.zero_grad()
        self.net_encoder_sourse.zero_grad()
        self.net_encoder_target.zero_grad()
        # self.net_sourse_mask.zero_grad()
        self.net_decoder.zero_grad()

        src_img = data_src['image'].to(self.device)
        obj_img = data_obj['image'].to(self.device)
        noobj_img = data_noobj['image'].to(self.device)
        # obj_only_img = data_obj_only.to(self.device)

        # src_img = torch.min(src_img, noobj_img)

        src_label = data_src['label'].to(self.device)

        batch_size = src_img.size(0)
        if self.label_src is None:
            l = np.array([self.src_label, self.obj_label, self.noobj_label])
            l_obj = np.random.choice(l, batch_size, p=[0.3, 0.7, 0])
            l_src = np.random.choice(l, batch_size, p=[0.7, 0.3, 0])
            l_noobj = np.random.choice(l, batch_size, p=[0.0, 0.3, 0.7])

            self.label_obj_f = torch.from_numpy(l_obj).float().to(self.device) 
            self.label_src_f = torch.from_numpy(l_src).float().to(self.device)
            self.label_noobj_f = torch.from_numpy(l_noobj).float().to(self.device) 

            self.label_obj = torch.full((batch_size,), self.obj_label, device=self.device) #torch.from_numpy(l_obj).float().to(self.device) #torch.full((batch_size,), self.obj_label, device=self.device)
            self.label_src =  torch.full((batch_size,), self.src_label, device=self.device) #torch.from_numpy(l_src).float().to(self.device) #torch.full((batch_size,), self.src_label, device=self.device)
            self.label_noobj =  torch.full((batch_size,), self.noobj_label, device=self.device) #torch.from_numpy(l_noobj).float().to(self.device) #torch.full((batch_size,), self.noobj_label, device=self.device)

        # self.net_encoder_sourse.eval()
        src_z, src_conv = self.net_encoder_sourse(src_img)
        # self.net_encoder_sourse.train()
        obj_z, obj_conv = self.net_encoder_sourse(obj_img)
        noobj_z, noobj_conv = self.net_encoder_sourse(noobj_img)

        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        errG_obj = self.get_loss_on_label(obj_conv, self.label_src, self.net_discriminator)
        errG_src = self.get_loss_on_label(src_conv, self.label_obj, self.net_discriminator)

        # obj_f, _ = self.net_encoder_sourse(obj_img)
        # noobj_f, _ = self.net_encoder_sourse(noobj_img)
        errG_obj_2 = self.get_loss_on_label(obj_conv, self.label_obj, self.net_discriminator_noobj)
        errG_noobj_2 = self.get_loss_on_label(noobj_conv, self.label_noobj, self.net_discriminator_noobj)

        ############################
        # (3) Update Src AutoED network: 
        ###########################
        src_autoed_loss = self.get_autoed_loss(self.net_encoder_sourse, src_z, src_label, noobj_z)
        self.update_generator(errG_obj, errG_src, errG_obj_2, errG_noobj_2, src_autoed_loss)


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        errD_src, errD_obj, errD2_obj, errD2_noobj  = self.get_discriminator_loss(src_conv, obj_conv, noobj_conv)
        self.update_discriminator(errD_src, errD_obj, errD2_obj, errD2_noobj)


        print('[%d][%d] Loss: %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f %.4f' % 
            (epoch, iteration, errD_src.item(), errD_obj.item(), errD2_obj.item(), errD2_noobj.item(), errG_obj.item(), 
                               errG_src.item(), errG_obj_2.item(), errG_noobj_2.item(), src_autoed_loss.item()))            


        # if self.count % 50 == 1:
        if iteration == 0:
            # for param_group in self.optimizer_autoed.param_groups:
            #     lr = param_group['lr']
            #     lr = lr - 0.0003/100
            #     if lr < 0.00001:
            #         lr = 0.00001
            #     param_group['lr'] = lr
            #     print(lr)
            self.set_eval_mode()
            # src_mu, src_conv = self.net_encoder_sourse.get_mu_conv(src_img)
            # src_reconst = self.net_decoder(src_mu)
            # src_reconst = self.decode_obj(src_conv)
            src_z, src_conv = self.net_encoder_sourse(src_img)
            src_mu, src_mu_conv = self.net_encoder_sourse.get_mu_conv(src_img)

            encoded_noobj, conv = self.net_encoder_target(noobj_img)
            randomized_z = torch.zeros_like(src_z)

            rand_w = 0.3 + np.random.rand()*0.3
            randomized_z = src_z * (1-rand_w) + encoded_noobj.detach()*rand_w

            # for i in range(len(src_z)):
            #     rand_w = 0.1 + np.random.rand()*0.9
            #     randomized_z[i] = src_z[i] * (1-rand_w) + encoded_noobj[i].detach()*rand_w

            src_reconst = self.net_decoder(randomized_z)


            # obj_mu, obj_conv = self.net_encoder_sourse.get_mu_conv(obj_img)
            # # obj_reconst = self.net_decoder(obj_mu)
            # obj_reconst = self.decode_obj(obj_conv) 
            obj_z, obj_conv = self.net_encoder_target(obj_img)
            obj_mu, obj_mu_conv = self.net_encoder_target.get_mu_conv(obj_img)
            obj_reconst = self.net_decoder(obj_mu)

            # noobj_mu, noobj_conv = self.net_encoder_sourse.get_mu_conv(noobj_img)
            # # noobj_reconst = self.net_decoder(noobj_mu)
            # noobj_reconst = self.decode_obj(noobj_conv)
            noobj_z, noobj_conv = self.net_encoder_target(noobj_img)
            noobj_mu, noobj_mu_conv = self.net_encoder_target.get_mu_conv(noobj_img)
            noobj_reconst = self.net_decoder(noobj_z)

            self.set_train_mode()

            size = src_conv[0].shape
            # src_conv = (src_conv + obj_conv.max() - noobj_conv)
            tool.save_result_imgs(outf, method, 
                            src_img = src_img, src_feature = (randomized_z).view(-1, 1, size[1], size[2]), 
                            src_reconst = src_reconst, src_conv = src_z.view(-1, 1, size[1], size[2]),
                            obj_img = obj_img, obj_feature = (obj_mu).view(-1, 1, size[1], size[2]), 
                            obj_reconst = obj_reconst, obj_conv = obj_z.view(-1, 1, size[1], size[2]),
                            noobj_img = noobj_img, noobj_feature = noobj_mu_conv, 
                            noobj_reconst = noobj_reconst, noobj_conv = noobj_conv)
            tool.save_models(outf, method, src_encoder = self.net_encoder_sourse, obj_encoder = self.net_encoder_target, decoder = self.net_decoder, 
                                                discriminator = self.net_discriminator, discriminator_noobj = self.net_discriminator_noobj)
            # torch.save(self.net_sourse_mask.state_dict(), 'results/%s/%s/net_mask.pth' % (outf, method))       
    
        # return errD_src, errD_obj, errD_obj, errG_obj, errG_noobj, src_autoed_loss 
        return src_autoed_loss.detach().item()