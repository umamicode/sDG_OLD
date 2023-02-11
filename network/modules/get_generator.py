from network import generator
from torch import optim
import torch

def get_generator(name, imdim=3, imsize= [32,32], lr= 1e-3, zdim=10, gen_mode=None):  
        if name=='cnn':
            #g1_noise= torch.randn(bs, zdim).cuda() #afterbreakup
            #g2_noise= torch.randn(bs, zdim).cuda() #afterbreakup
            g1_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g2_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif name=='hr':
            raise ValueError("HR Generator is not used nor implemented in our model. (Trace of PDEN code)")
        elif name=='stn':
            g1_net = generator.stnGenerator(zdim=zdim,imsize=imsize, mode=gen_mode).cuda()
            g2_net = None
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr/2)
            g2_opt = None
        return g1_net, g2_net, g1_opt, g2_opt