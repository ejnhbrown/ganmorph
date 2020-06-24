import random
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Model_Classes import Generator, Discriminator
from Training_funcs import weights_init, training_loop
from pickle import dump
import os

###################
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
###################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataroot = "data/wikiartsmall"
model_name = '200_features'

workers = 0  # Number of workers for dataloader
batch_size = 64  # Batch size during training
image_size = 128  # Spatial size of training images. All images will be resized to this size using a transformer.
nc = 3  # Number of channels in the training images. For color images this is 3
nz = 200  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 7  # Number of training epochs
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ______TRAINING SETUP_______
results_dir = 'Results/'+model_name+'/'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = Generator(ngpu, nz, ngf, nc).to(device)  # Create the generator
netG.apply(weights_init)  # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
print(netG)

netD = Discriminator(ngpu, ndf, nc).to(device)  # Create the Discriminator
netD.apply(weights_init)  # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
print(netD)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^


# ______TRAIN______
D_losses, G_losses, img_list = training_loop(results_dir, num_epochs, dataloader, netD, netG, device, lr, beta1, nz)
dump(img_list, open(results_dir+'img_list.pkl','wb'))
torch.save(netD, results_dir+'final_disc.pyt')
torch.save(netG, results_dir+'final_gen.pyt')
# ^^^^^^^^^^^^^^^^^


# ______PLOTTING____
################################################################################################

# Losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(results_dir+"Loss_Curve.png")
plt.show()

# Noise vector animation
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, bitrate=1800)
ani.save(results_dir+"ganmorph.mp4", writer=writer)
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.savefig(results_dir+"Real_Fake_Comparison.png")

################################################################################################

plt.show()