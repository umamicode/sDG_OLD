from network import generator
from torch import optim

def get_generator(name, imdim=3, imsize= [32,32], lr= 1e-3):  
        if name=='cnn':
            g1_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g2_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif gen=='hr':
            raise ValueError("HR Generator is not Implemented in our model.")
            1/0
            g1_net = hrnet.HRGenerator(zdim=zdim).cuda()
            g2_net = hrnet.HRGenerator(zdim=zdim).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif gen=='stn':
            g1_net = generator.stnGenerator(zdim=zdim, mode=gen_mode).cuda()
            g2_net = None
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr/2)
            g2_opt = None
        return g1_net, g2_net, g1_opt, g2_opt