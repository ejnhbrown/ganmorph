import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess as sp
import torchvision.utils as vutils
import time
from pickle import load, dump
from PIL import Image
from PIL import ImageEnhance
from torch import tensor
from torch import load as tload
from tqdm import tqdm
from matplotlib.offsetbox import AnnotationBbox
import os
#from ISR.models import RRDN

model = '200_features_cont'
results_dir = 'Results/'+model+'/'
fps = 6
#GRID IS (X,Y) = (16,8)

def foo(imlist, num, x, y):
    return np.transpose(imlist[num][:, (130 * x) + 1:(130 * (x + 1)) + 1, (130 * y) + 1:(130 * (y + 1)) + 1])

def ffmpeg_pil(filepath,ims,fps,res):

    p = sp.Popen(
        ['ffmpeg', '-y',
         '-f', 'image2pipe',
         '-vcodec', 'png',
         '-s', str(res)+'x'+str(res),
         '-r', str(fps),
         '-i', '-',
         '-an',
         '-vcodec', 'mpeg4',
         '-qscale','1',
         '-r', '24',
         filepath], stdin=sp.PIPE)

    for s_i, subim in enumerate(ims):
        print(s_i)
        subim.save(p.stdin, 'PNG')

    p.stdin.close()
    p.wait()

def animate_one(results_dir, x, y,fps=2,bitrate=1800):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    imlist = load(open(results_dir+'image_list.pkl', 'rb'))
    ims = [[plt.imshow(foo(imlist,i,x,y))] for i in range(len(imlist))]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=bitrate)
    ani.save("vids/"+model+".mp4", writer=writer)
    plt.show()

def animate_one_upscaled_highcont(results_dir, x, y,fps=2,res=500,cont=1.5):

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    imlist = load(open(results_dir + 'image_list.pkl', 'rb'))
    ims = [Image.fromarray( (np.array(foo(imlist, i, x, y))*255).astype(np.uint8) ).resize(res)
           for i in range(len(imlist))]
    ims = [ImageEnhance.Contrast(i).enhance(cont) for i in ims]

    ffmpeg_pil('ffvid.avi',ims,fps,res)

def progress_grid(results_dir,fps=10,bitrate=3600):
    # Noise vector animation
    img_list = load(open('Results/'+ results_dir + '/image_list.pkl', 'rb'))
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ax = plt.axes()
    ims = [[plt.imshow(np.transpose(thisim, (1, 2, 0)), animated=True),ax.annotate('Model: '+results_dir+'\nSnapshot: '+str(i),(50,150),color='r')] for i,thisim in enumerate(img_list)]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=bitrate)
    ani.save('Results/'+ results_dir + "/custom_ganmorph.mp4", writer=writer)
    plt.show()


# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return np.asarray(vectors)


def meander(results_dir,n_locs,interp_points,latent_dim=100,contrast=1.0,bright=1.0,sat=1.0,fps = 24,loop=True):

    if not loop:
        x_input = np.random.randn(latent_dim * n_locs)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_locs, latent_dim)

    else:
        x_input = np.random.randn(latent_dim * (n_locs-1))
        # reshape into a batch of inputs for the network
        x_input = np.append(x_input,x_input[:latent_dim])
        z_input = x_input.reshape(n_locs, latent_dim)

    reslist = []
    for i in range(len(z_input) - 1):
        reslist.append(interpolate_points(z_input[i], z_input[i + 1], interp_points))

    res = np.concatenate(reslist)
    resten = tensor(res.reshape((res.shape[0],res.shape[1],1,1))).float()

    modfile = 'Results\\' + results_dir + '\generator.pyt'
    netG = tload(modfile, map_location=lambda storage, loc: storage)

    fake = netG(resten).detach().cpu()
    fakearr = np.array(fake)

    dump(fakearr,open('../fakearr.pkl','wb'))


    ims = [Image.fromarray((((fakearr[i]+1)/2)*255).astype(np.uint8).transpose().reshape((128,128,3)))\
                for i in range(len(fakearr))]
    ims = [ImageEnhance.Contrast(i).enhance(contrast) for i in ims]
    ims = [ImageEnhance.Color(i).enhance(sat) for i in ims]
    ims = [ImageEnhance.Brightness(i).enhance(bright) for i in ims]

    if not os.path.isdir('upsc_ims/'):
        os.mkdir('upsc_ims/')

    #######
    alph = 'abcdefghijklmnopqrstuvwxyz'
    alphlist = [alph[i] + alph[j] + alph[k] for i in range(26) for j in range(26) for k in range(26)]
    for im_i, img in tqdm(enumerate(ims)):
        img.save('upsc_ims/im_'+alphlist[im_i]+'.png')
    #######



    ffmpeg_pil('Results/'+results_dir+'/meander.mp4',ims,fps=fps,res=ims[0].size[0])


if __name__ == '__main__':

    #animate_one(results_dir,6,7,fps=fps)

    #animate_one_upscaled_highcont(results_dir,11,4,fps=fps,cont=1.5)

    #meander('200_features_cont5',15,30,fps=30,latent_dim=200,contrast=3.7 ,bright=1.1)

    progress_grid('200_features_cont7',fps=4)