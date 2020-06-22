import torch.nn as nn
from torch import load, device, randn, cuda
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


nc = 3
nz = 100
ngf = 64
ngpu = 1
modfile = 'models/gens/128,100,5ep_gen.pyt'

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu)
netG = load(modfile)
#netG.load_state_dict(load(modfile))

print(1)
device = device("cuda:0" if (cuda.is_available() and ngpu > 0) else "cpu")
print(2)
fixed_noise = randn(64, nz, 1, 1, device=device)
print(3)
fake = netG(fixed_noise).detach().cpu()
print(4)

v = vutils.make_grid(fake, padding=2, normalize=True)

plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(v))
plt.show()

print(5)