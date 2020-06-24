import torch.nn as nn
from torch import load, device, randn, cuda
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from Model_Classes import Generator


nc = 3
nz = 100
ngf = 64
ngpu = 1
modfile = r'Results\128_10Ep_onwards\generator.pyt'


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