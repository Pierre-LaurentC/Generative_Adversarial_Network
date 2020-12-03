import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

try:
    filesXtrain = os.listdir('../GAN/DCGAN/images/')
    numberOfFiles = len(filesXtrain)
except: print('File not found')
numberOfFiles-=1

matrices = np.empty((numberOfFiles,10,24,144))

for i in range(numberOfFiles):
    matrices[i] = np.load('../GAN/DCGAN/images/DCGAN_prediction_{}.npy'.format(i))

numberOfEpochs = numberOfFiles
fig, axs = plt.subplots(5,2, figsize=(25, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)

axs = axs.ravel().reshape(5,2)
for y in range(numberOfEpochs):
    m_index=0
    for t in range(5):
        axs[t][0].imshow(matrices[y][m_index], origin='lower', cmap=plt.get_cmap('jet', 20) , aspect='auto')
        axs[t][1].imshow(matrices[y][m_index+1], origin='lower', cmap=plt.get_cmap('jet', 20) , aspect='auto')
        if m_index<8: m_index+=2
    fig.savefig("../GAN/DCGAN/imagesForGif/image_{}".format(y))
    
frames = []
for image in range(numberOfFiles):
    frames.append(Image.open('DCGAN/imagesForGif/image_{}.png'.format(image)))
frames[0].save('gan_training.gif', format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)