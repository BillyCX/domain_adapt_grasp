import torchvision.utils as vutils
import torch

normalize = True

def save_test_imgs(outf, method, test_img=None, test_feature=None, test_conv=None, ext=''):
    if test_img is not None:
        vutils.save_image(test_img,
        'results/%s/%s/%stest_img.png' % (outf, method, ext),
        normalize=normalize)
    if test_feature is not None:
        vutils.save_image(test_feature,
        'results/%s/%s/%stest_feature.png' % (outf, method, ext),
        normalize=normalize)
    if test_conv is not None:
        vutils.save_image(test_conv,
        'results/%s/%s/%stest_conv.png' % (outf, method, ext),
        normalize=normalize)    

def save_result_imgs(outf, method, 
                    src_img = None, src_feature = None, src_reconst = None, src_conv = None,
                    obj_img = None, obj_feature = None, obj_reconst = None, obj_conv = None,
                    noobj_img = None, noobj_feature = None, noobj_reconst = None, noobj_conv = None):
    if src_img is not None:
        vutils.save_image(src_img,
        'results/%s/%s/src_img.png' % (outf, method),
        normalize=normalize)
    if src_feature is not None:
        vutils.save_image(src_feature,
        'results/%s/%s/src_feature.png' % (outf, method),
        normalize=normalize)
    if src_reconst is not None:
        vutils.save_image(src_reconst,
        'results/%s/%s/src_reconst.png' % (outf, method),
        normalize=normalize)
    if src_conv is not None:
        vutils.save_image(src_conv,
        'results/%s/%s/src_conv.png' % (outf, method),
        normalize=normalize)

    if obj_img is not None:
        vutils.save_image(obj_img,
        'results/%s/%s/obj_img.png' % (outf, method),
        normalize=normalize)
    if obj_feature is not None:
        vutils.save_image(obj_feature,
        'results/%s/%s/obj_feature.png' % (outf, method),
        normalize=normalize)
    if obj_reconst is not None:
        vutils.save_image(obj_reconst,
        'results/%s/%s/obj_reconst.png' % (outf, method),
        normalize=normalize)
    if obj_conv is not None:
        vutils.save_image(obj_conv,
        'results/%s/%s/obj_conv.png' % (outf, method),
        normalize=normalize)

    if noobj_img is not None:
        vutils.save_image(noobj_img,
        'results/%s/%s/noobj_img.png' % (outf, method),
        normalize=normalize)
    if noobj_feature is not None:
        vutils.save_image(noobj_feature,
        'results/%s/%s/noobj_feature.png' % (outf, method),
        normalize=normalize)
    if noobj_reconst is not None:
        vutils.save_image(noobj_reconst,
        'results/%s/%s/noobj_reconst.png' % (outf, method),
        normalize=normalize)
    if noobj_conv is not None:
        vutils.save_image(noobj_conv,
        'results/%s/%s/noobj_conv.png' % (outf, method),
        normalize=normalize)
                

def save_models(outf, method, src_encoder = None, obj_encoder = None, decoder = None, discriminator = None, discriminator_noobj = None):
    if src_encoder is not None:
        torch.save(src_encoder.state_dict(), 'results/%s/%s/net_se.pth' % (outf, method))
    if obj_encoder is not None:
        torch.save(obj_encoder.state_dict(), 'results/%s/%s/net_te.pth' % (outf, method))    
    if decoder is not None:
        torch.save(decoder.state_dict(), 'results/%s/%s/net_d.pth' % (outf, method))
    if discriminator is not None:
        torch.save(discriminator.state_dict(), 'results/%s/%s/net_dis.pth' % (outf, method))
    if discriminator_noobj is not None:
        torch.save(discriminator_noobj.state_dict(), 'results/%s/%s/net_dis_noobj.pth' % (outf, method))         

def load_models(outf, method, src_encoder = None, obj_encoder = None, decoder = None, discriminator = None, discriminator_noobj = None):
    if src_encoder is not None:
        src_encoder.load_state_dict(torch.load('results/%s/%s/net_se.pth' % (outf, method)))
    if obj_encoder is not None:
        obj_encoder.load_state_dict(torch.load('results/%s/%s/net_te.pth' % (outf, method)))  
    if decoder is not None:
        decoder.load_state_dict(torch.load('results/%s/%s/net_d.pth' % (outf, method)))
    if discriminator is not None:
        discriminator.load_state_dict(torch.load('results/%s/%s/net_dis.pth' % (outf, method))) 
    if discriminator_noobj is not None:
        discriminator_noobj.load_state_dict(torch.load('results/%s/%s/net_dis_noobj.pth' % (outf, method)))         