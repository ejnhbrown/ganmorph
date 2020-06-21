from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

######
dir = 'data/wikiart/'
newdir = 'data/wikiartsmall/'
######
output_res = (128,128)


dirlist = os.listdir(dir)
dirlen = len(dirlist)

for folder_i in range(dirlen):
    folder = dirlist[folder_i]
    loc = dir+folder + '/'
    newloc = newdir+folder + '/'
    if not os.path.isdir(newloc):
        os.mkdir(newloc)

    print('Processing Folder:',folder,' |',str(folder_i+1)+'/'+str(dirlen))
    imlocs = os.listdir(loc)

    for img in tqdm(imlocs):

        Im = Image.open(loc+img)
        mindim = min(Im.width, Im.height)
        cropim = Im.crop((0, 0, mindim, mindim))
        resim = cropim.resize(output_res, Image.ANTIALIAS)
        resim.save(newloc + img + '.jpg')