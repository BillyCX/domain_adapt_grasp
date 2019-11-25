from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
real_label = 1
fake_label = 0
n_z = 3
beta = 0
beta_scale = 1
vae_rec_loss = []
epoch_rec_loss, epoch_kld_loss = [], []

class EncoderSourse(nn.Module):
    def __init__(self, ngpu=1):
        super(EncoderSourse, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),   # 256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #128 

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # 128
            nn.BatchNorm2d(64),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),  # 64

            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 32

            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),         

            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),   

            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(16),
            nn.ReLU(True),     

            nn.Conv2d(16, 1, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(1),
            nn.ReLU(True),                      
        )
        self.fc1 = nn.Linear(1*16*16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_z)
        self.fc4 = nn.Linear(128, n_z)
        self.actf = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().float().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).float().cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        fc1 = self.actf(self.fc1(h))
        fc2 = self.actf(self.fc2(fc1))

        mu, logvar = self.fc3(fc2), self.fc4(fc2)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, input):
        h = self.main(input)
        z, mu, logvar = self.bottleneck(h.view(-1, 1*16*16))

        return z, mu, logvar, h

class EncoderTarget(nn.Module):
    def __init__(self, ngpu=1):
        super(EncoderTarget, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),   # 256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #128 

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # 128
            nn.BatchNorm2d(64),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),  # 64

            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 32

            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(256),
            nn.ReLU(True),         

            nn.Conv2d(256, 64, 3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),                 
        )
        self.fc1 = nn.Linear(64*16*16, 64*16*16)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_z)
        self.fc4 = nn.Linear(512, n_z)

    def forward(self, input):
        h = self.main(input)
        # feature = self.fc1(h.view(-1, 64*16*16))
        return h

class Decoder(nn.Module):
    def __init__(self, ngpu=1):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(n_z, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True), 
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),   
            nn.Linear(1024, 1*128*128),
            nn.Tanh()
                                   
            # nn.ConvTranspose2d(16, 32, 3, stride=2),  # 17
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),  # 33
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),  # 64
            # nn.Tanh()
        )
        # self.fc1 = nn.Linear(n_z, 128*32*32)   

    def forward(self, input):
        # h = self.fc1(input)
        # output = self.main(h.view(-1, 16, 8, 8))
        output = self.main(input)
        return output.view(-1, 1, 128, 128)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.Linear(n_z, n_z*2),
            # nn.BatchNorm1d(n_z*2),
            # nn.ReLU(True), 

            # nn.Linear(n_z*2, n_z*4),
            # nn.BatchNorm1d(n_z*4),
            # nn.ReLU(True),   

            # nn.Linear(n_z*4, n_z*8),
            # nn.BatchNorm1d(n_z*8),
            # nn.ReLU(True),  

            # nn.Linear(n_z*8, 1),
            # nn.Sigmoid()

            # nn.Conv2d(3, 32, 3, stride=1, padding=1),   # 256
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), #128 

            # nn.Conv2d(32, 64, 3, stride=1, padding=1),  # 128
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),            
            # nn.MaxPool2d(2, stride=2),  # 64

            # nn.Conv2d(64, 32, 3, stride=1, padding=1),  # 64
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2)  # 32

            # input is (nc) x 8 x 8
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(32, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()           

            # nn.Linear(1*16*16, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True), 
            # nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),   
            # nn.Linear(64, 1),
            # nn.Sigmoid()                
        )
        self.fc1 = nn.Linear(16*16*16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        # # # print(output.shape)
        fc1 = self.relu(self.fc1(output.view(-1, 16*16*16)))
        fc2 = self.relu(self.fc2(fc1))
        fc3 = self.sigmoid(self.fc3(fc2))

        return fc3.view(-1, 1).squeeze(1)

def save_result_imgs(outf, method, 
                     src_img = None, src_feature = None, src_reconst = None, src_conv = None,
                     obj_img = None, obj_feature = None, obj_reconst = None, obj_conv = None):
    if src_img is not None:
        vutils.save_image(src_img,
            'results/%s/%s/src_img.png' % (outf, method),
            normalize=True)
    if src_feature is not None:
        vutils.save_image(src_feature,
            'results/%s/%s/src_feature.png' % (outf, method),
            normalize=True)
    if src_reconst is not None:
        vutils.save_image(src_reconst,
            'results/%s/%s/src_reconst.png' % (outf, method),
            normalize=True)
    if src_conv is not None:
        vutils.save_image(src_conv,
            'results/%s/%s/src_conv.png' % (outf, method),
            normalize=True)

    if obj_img is not None:
        vutils.save_image(obj_img,
            'results/%s/%s/obj_img.png' % (outf, method),
            normalize=True)
    if obj_feature is not None:
        vutils.save_image(obj_feature,
            'results/%s/%s/obj_feature.png' % (outf, method),
            normalize=True)
    if obj_reconst is not None:
        vutils.save_image(obj_reconst,
            'results/%s/%s/obj_reconst.png' % (outf, method),
            normalize=True)
    if obj_conv is not None:
        vutils.save_image(obj_conv,
            'results/%s/%s/obj_conv.png' % (outf, method),
            normalize=True)                                               

def load_models(outf, method):
    global net_encoder_sourse, net_decoder
    net_encoder_sourse.load_state_dict(torch.load('results/%s/%s/net_se.pth' % (outf, method)))
    net_decoder.load_state_dict(torch.load('results/%s/%s/net_d.pth' % (outf, method)))

def get_feature_src(data):
    img = data.to(device)
    encoded_f, mu, logvar, conv = net_encoder_sourse(img)
    return mu, conv

def get_feature_obj(data):
    img = data.to(device)
    conv = net_encoder_target(img)
    return conv

def decode_obj(obj_conv):
    # obj_mu = net_encoder_sourse.fc1(obj_conv.view(-1, 16*8*8))
    obj_z, obj_mu, _ = net_encoder_sourse.bottleneck(obj_conv.view(-1, 1*16*16))  
    # print(obj_mu.shape) 
    obj_decoded = net_decoder(obj_mu)
    return obj_z, obj_mu, obj_decoded

def loss_function(recon_x, x, mu, logvar, beta):
    BCE = criterion_autoed(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD_batch)

    print(BCE.item(), KLD.item(), beta)
    # print(torch.mean(logvar.exp(), dim=0), torch.mean(mu, dim=0))
    return BCE, KLD, BCE + KLD*0.0001



def update_domain_adapt(data_src, data_obj, iteration, outf, method):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    net_discriminator.zero_grad()
    src_img = data_src.to(device)

    batch_size = src_img.size(0)
    label = torch.full((batch_size,), real_label, device=device)
    label.normal_(real_label, 0.02)

    src_feature, src_mu, src_logvar, src_conv = net_encoder_sourse(src_img)
    src_disc = net_discriminator(src_conv.detach())
    # src_disc = net_discriminator(src_img)

    errD_src = criterion_adv(src_disc, label)
    errD_src.backward()
    D_x = src_disc.mean().item()

    # train with fake
    obj_img = data_obj.to(device)
    label.normal_(fake_label, 0.02)
    # label.fill_(fake_label)

    # obj_conv = net_encoder_target(obj_img)
    _, _, _, obj_conv = net_encoder_sourse(obj_img)    
    obj_z, obj_mu, obj_reconst = decode_obj(obj_conv)
    obj_disc_d = net_discriminator(obj_conv.detach())
    # obj_disc_d = net_discriminator(obj_reconst)

    errD_obj = criterion_adv(obj_disc_d, label)
    errD_obj.backward()
    D_G_z1 = obj_disc_d.mean().item()
    errD = errD_src + errD_obj

    if errD > 1:
        optimizer_d.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    net_encoder_sourse.zero_grad()
    # net_encoder_target.zero_grad()
    # net_decoder_target.zero_grad()

    label.fill_(real_label)  # fake labels are real for generator cost
    # label.normal_(real_label, 0.01)

    _, obj_mu, _, obj_conv = net_encoder_sourse(obj_img)
    obj_reconst = net_decoder(obj_mu)

    # obj_conv = net_encoder_target(obj_img)
    # obj_z, obj_mu, obj_reconst = decode_obj(obj_conv)

    obj_disc_g = net_discriminator(obj_conv)
    # obj_disc_g = net_discriminator(obj_reconst)
    obj_errG = criterion_adv(obj_disc_g, label)
    
    errG = obj_errG
    errG.backward()
    D_G_z2 = obj_disc_g.mean().item()
    # if not update_D:
    optimizer_et.step()

    if iteration == 0:
        # obj_reconst = decode_obj(obj_conv)
        src_reconst = net_decoder(src_mu)
        save_result_imgs(outf, method, 
                        src_img = src_img, src_feature = None, src_reconst = src_reconst, src_conv = src_conv[0].view(1, 1, 16, 16),
                        obj_img = obj_img, obj_feature = None, obj_reconst = obj_reconst, obj_conv = obj_conv[0].view(1, 1, 16, 16))
        torch.save(net_encoder_target.state_dict(), 'results/%s/%s/net_te.pth' % (outf, method))
        torch.save(net_discriminator.state_dict(), 'results/%s/%s/net_dis.pth' % (outf, method))  

    return errD, errG


def update_src_autoed(data, iteration, outf, method):
    global beta, beta_scale, epoch_rec_loss, epoch_kld_loss, vae_rec_loss
    net_encoder_sourse.zero_grad()
    net_decoder.zero_grad()

    img = data.to(device)

    encoded_f, mu, logvar, conv = net_encoder_sourse(img)

    # print('mu', encoded_f[0], mu[0])
    # print('mu', encoded_f[0])
    output = net_decoder(encoded_f)
    rec_err, kld, err = loss_function(output, img, mu, logvar, beta) #criterion_autoed(output, img)
    
    epoch_rec_loss.append(rec_err.item())
    epoch_kld_loss.append(kld.item())

    # output_mu = net_decoder(mu)
    err.backward()

    optimizer_autoed.step()
    # print(conv.shape)

    if iteration == 0:
        epoch_rec_loss_mean = np.mean(epoch_rec_loss)
        epoch_kld_loss_mean = np.mean(epoch_kld_loss)

        vae_rec_loss.append(epoch_rec_loss_mean)
        epoch_rec_loss, epoch_kld_loss = [], []

        print('   ********* mean loss', epoch_rec_loss_mean, epoch_kld_loss_mean, 'beta', beta)

        # if epoch_rec_loss_mean < 0.003:
        #     # beta += 0.001
        #     beta_scale += 0.01
        #     beta = epoch_rec_loss_mean/epoch_kld_loss_mean * beta_scale
        #     print('update beta', beta, beta_scale)
        # else:
        #     beta_scale -= 0.01
        # beta_scale = np.clip(beta_scale, 0, 5)

        plt.clf()
        plt.plot(np.log(np.asarray(vae_rec_loss)))
        plt.savefig('results/%s/%s/rec_loss.png' % (outf, method))
        np.save('results/%s/%s/rec_loss' % (outf, method), vae_rec_loss)
        output_mu = net_decoder(mu)
        save_result_imgs(outf, method, 
                     src_img = img, src_feature = output_mu, src_reconst = output, src_conv = conv[0].view(1, 1, 16, 16),
                     obj_img = None, obj_feature = None, obj_reconst = None, obj_conv = None)
        torch.save(net_encoder_sourse.state_dict(), 'results/%s/%s/net_se.pth' % (outf, method))
        torch.save(net_decoder.state_dict(), 'results/%s/%s/net_d.pth' % (outf, method))

    return err

# create networks
lr = 0.0002
device = torch.device("cuda:0") #torch.device("cuda:0" if opt.cuda else "cpu")                                    EdgeCrop(130, 70, 30, 60),


net_encoder_sourse = EncoderSourse().to(device)
print(net_encoder_sourse)
net_encoder_target = EncoderTarget().to(device)
print(net_encoder_target)
net_decoder = Decoder().to(device)
print(net_decoder)
# net_decoder_target = DecoderTarget().to(device)
# print(net_decoder_target)
net_discriminator = Discriminator().to(device)
print(net_discriminator)

# autoencoder loss
criterion_autoed = nn.MSELoss()
autoencoder_params = list(net_encoder_sourse.parameters()) + list(net_decoder.parameters())
optimizer_autoed = torch.optim.Adam(autoencoder_params, lr=lr)

# target encoder loss
targetencoder_params = list(net_encoder_target.parameters())
# targetencoder_params = list(net_encoder_target.parameters()) + list(net_decoder_target.parameters())
optimizer_et = torch.optim.Adam(targetencoder_params, lr=lr)

# discriminator loss
criterion_adv = nn.BCELoss()
discriminator_params = list(net_discriminator.parameters())
optimizer_d = torch.optim.Adam(discriminator_params, lr=lr)

