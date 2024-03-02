import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .base import Base, BaseTrain
from ..networks import ADN, NLayerDiscriminator, add_gan_loss
from ..utils import print_model, get_device
from scipy.io import savemat
import os

save_path = '/data/tt/semimar_deep_nocyc'
class ContrastLoss(nn.Module):
    def __init__(self):

        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = a, p, n
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contrastive = d_ap / (d_an + 1e-7)

            loss += contrastive
        return loss
class ADNTrain(BaseTrain):
    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(ADNTrain, self).__init__(learn_opts, loss_opts)
        self.t = 0
        g_opts, d_opts = model_opts[g_type], model_opts[d_type]

        model_dict = dict(
              adn = lambda: ADN(**g_opts),
            nlayer = lambda: NLayerDiscriminator(**d_opts))

        self.model_g = self._get_trainer(model_dict, g_type) # ADN generators
        self.model_dl = add_gan_loss(self._get_trainer(model_dict, d_type)) # discriminator for low quality image (with artifact)
        self.model_dh = add_gan_loss(self._get_trainer(model_dict, d_type)) # discriminator for high quality image (without artifact)
        # self.model_dA = add_gan_loss(self._get_trainer(model_dict, d_type)) # discriminator for low quality image (with artifact)
        # self.model_dB = add_gan_loss(self._get_trainer(model_dict, d_type)) # discriminator for high quality image (without artifact)

        loss_dict = dict(
               l1 = nn.L1Loss,
               con = ContrastLoss,
               gl = (self.model_dl.get_g_loss, self.model_dl.get_d_loss), # GAN loss for low quality image.
               gh = (self.model_dh.get_g_loss, self.model_dh.get_d_loss)) # GAN loss for high quality image

        # Create criterion for different loss types
        self.model_g._criterion["ll"] = self._get_criterion(loss_dict, self.wgts["ll"], "ll_")
        self.model_g._criterion["hh"] = self._get_criterion(loss_dict, self.wgts["hh"], "hh_")
        self.model_g._criterion["lh"] = self._get_criterion(loss_dict, self.wgts["lh"], "lh_")
        self.model_g._criterion["lhl"] = self._get_criterion(loss_dict, self.wgts["lhl"], "lhl_")
        self.model_g._criterion["hlh"] = self._get_criterion(loss_dict, self.wgts["hlh"], "hlh_")
        self.model_g._criterion["noise"] = self._get_criterion(loss_dict, self.wgts["noise"], "noise_")
        self.model_g._criterion["gl"] = self._get_criterion(loss_dict, self.wgts["gl"])
        self.model_g._criterion["gh"] = self._get_criterion(loss_dict, self.wgts["gh"])
        self.model_g._criterion["cont"] = self._get_criterion(loss_dict, self.wgts["cont"], "cont")
        self.model_g._criterion["contdeep"] = self._get_criterion(loss_dict, self.wgts["contdeep"], "contdeep_")

    def _nonzero_weight(self, *names):
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if type(w[0]) is str: w = [w]
            for p in w: wgt += p[1]
        return wgt

    def optimize(self, name, A, B, C):
        self.x_low, self.x_high, self.pre_CT = self._match_device(A, B, C)
        self.model_g._clear()
        self.l_h,self.h_l,self.h_h,self.l_h_l,self.h_l_h,self.noise1,self.noise2,self._negtive,self._negtive1,self._positive,self._anchor = self.model_g.forward1(self.x_low, self.x_high, self.pre_CT)
# l_h,h_l,h_h,l_h_l,h_l_h,noise1,noise2
        # low -> low_l, low -> low_h
        # for iii in range(12):
        #    savemat(os.path.join(save_path,'out%06d' % (self.t+1)+'.mat'), {'out':np.array(torch.squeeze(self.l_h[iii]).cpu().detach().numpy() )})

        #    self.t+=1

        self.negtive_shallow = self._negtive[0]
        self.positive_shallow = self._positive[0]
        self.anchor_shallow = self._anchor[0]
        # self.negtive_shallow = self._negtive[:2]
        # self.positive_shallow = self._positive[:2]
        # self.anchor_shallow = self._anchor[:2]
        
        if self._nonzero_weight("gl", "lh", 'hh'):
            self.model_dl._clear()
            self.model_g._criterion["gl"](self.l_h, self.x_high)
            self.model_g._criterion["lh"](self.l_h, self.pre_CT)
            self.model_g._criterion["hh"](self.h_h, self.x_high)
            self.model_g._criterion["cont"](self.anchor_shallow, self.positive_shallow, self.negtive_shallow)
            # self.model_g._criterion["contdeep"](self.anchor_deep, self.positive_deep, self.negtive_deep)

        # high -> high_l, high -> high_h
        if self._nonzero_weight("gh"):
            self.model_dh._clear()
            self.model_g._criterion["gh"](self.h_l, self.x_low)

        # low_h -> low_h_l
        if self._nonzero_weight("lhl"):
            self.model_g._criterion["lhl"](self.l_h_l, self.x_low)

        # high_l -> high_l_h
        if self._nonzero_weight("hlh"):
            self.model_g._criterion["hlh"](self.h_l_h, self.x_high)

        # artifact
        if self._nonzero_weight("noise"):
            self.model_g._criterion["noise"](self.noise1,self.noise2)
       


        
        self.model_g._update()
        self.model_dl._update()
        self.model_dh._update()
        
        

        # merge losses for printing
        self.loss = self._merge_loss(
            self.model_dl._loss, self.model_dh._loss, self.model_g._loss)

    def get_visuals(self, n=12):
        lookup = [
            ("l", "x_low"), ("pre", "pre_CT"), ("lh", "l_h"), ("lhl", "l_h_l"),
            ("h", "x_high"), ("hl", "h_l"), ("hh", "h_h"), ("hlh", "h_l_h")]

        return self._get_visuals(lookup, n)

    def evaluate(self, loader, metrics):
        progress = tqdm(loader)
        res = defaultdict(lambda: defaultdict(float))
        cnt = 0
        for img_low, img_high in progress:
            img_low, img_high = self._match_device(img_low, img_high)

            def to_numpy(*data):
                data = [loader.dataset.to_numpy(d, False) for d in data]
                return data[0] if len(data) == 1 else data

            pred_ll, pred_lh = self.model_g.forward1(img_low)
            pred_hl, pred_hh = self.model_g.forward2(img_low, img_high)
            pred_hlh = self.model_g.forward_lh(pred_hl)
            img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh = to_numpy(
                img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh)

            met = {
                "ll": metrics(img_low, pred_ll),
                "lh": metrics(img_high, pred_lh),
                "hl": metrics(img_low, pred_hl),
                "hh": metrics(img_high, pred_hh),
                "hlh":metrics(img_high, pred_hlh)}

            res = {n: {k: (res[n][k] * cnt + v) / (cnt + 1) for k, v in met[n].items()} for n in met}
            desc = "[{}]".format("/".join(met["ll"].keys()))
            for n, met in res.items():
                vals = "/".join(("{:.2f}".format(v) for v in met.values()))
                desc += " {}: {}".format(n, vals)
            progress.set_description(desc=desc)


class ADNTest(Base):
    def __init__(self, g_type, **model_opts):
        super(ADNTest, self).__init__()
        self.t = 0

        g_opts = model_opts[g_type]
        model_dict = dict(adn = lambda: ADN(**g_opts))
        self.model_g = model_dict[g_type]()
        print_model(self)

    def forward(self, img_low):
        self.img_low = self._match_device(img_low)
        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)

        return  self.pred_ll, self.pred_lh

    def evaluate(self, name, A, B, C):
        self.A, self.B, self.C = self._match_device(A, B, C)
        self.name = name
        # self.l_h,self.h_l,self.h_h,self.l_h_l,self.h_l_h,self.noise1,self.noise2 = self.model_g.forward1(self.A, self.B)
        self.l_h = self.model_g.forward2(self.A)
        for iii in range(20):
           savemat(os.path.join(save_path,'out%06d' % (self.t+1)+'.mat'), {'out':np.array(torch.squeeze(self.l_h[iii]).cpu().detach().numpy() )})

           self.t+=1

        # self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
        # self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
        # self.pred_hlh = self.model_g.forward_lh(self.pred_hl)

    def get_pairs(self):
        return [
            ("before", (self.img_low, self.img_high)), 
            ("after", (self.pred_lh, self.img_high))], self.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh"),
            ("cleanA", "cleanA"),("cleanB", "cleanB")]
        func = lambda x: x * 0.5 + 0.5
        return self._get_visuals(lookup, n, func, False)
