import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import *


'''
img = np.random.rand(1, 500, 500, 3)
fig, ax = plt.subplots(1, 5, figsize=(20, 10))
ax = ax.ravel()
ax[0].imshow(img[0])
ax[1].imshow(next(ImageDataGenerator().flow(img))[0])
ax[2].imshow(next(ImageDataGenerator(brightness_range=(0., 0.)).flow(img))[0])
ax[3].imshow(next(ImageDataGenerator(brightness_range=(1., 1.)).flow(img))[0])
ax[4].imshow(next(ImageDataGenerator(brightness_range=(1., 1.)).flow(img))[0] / 255)
plt.show()
'''


fname = '/Users/wang/Desktop/solar-pv-in-aerial-imagery/data/training/4.tif'
img = matplotlib.image.imread(fname)


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


'''
fig, axs = plt.subplots(2,2)
ax = axs.flatten()
ax[0].imshow(img)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title("Original Image")
ax[1].imshow(next(ImageDataGenerator(shear_range = 0.2).flow(img.reshape(1,101,101,3)))[0] / 255)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Shear Range 0.2")
ax[2].imshow(next(ImageDataGenerator(zoom_range = 0.2).flow(img.reshape(1,101,101,3)))[0] / 255)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title("Zoom Range 0.2")
ax[3].imshow(next(ImageDataGenerator(horizontal_flip = True).flow(img.reshape(1,101,101,3)))[0] / 255)
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[3].set_title("Horizontal Flipped")
plt.tight_layout()
plt.show()
'''

plt.imshow(img)
plt.title("Original Image")
plt.xticks([])
plt.yticks([])
plt.show()


datagen = ImageDataGenerator(shear_range = 10)
fig, axs = plt.subplots(2,2, figsize=(8,8))
ax = axs.flatten()
for a in ax:
    a.imshow(next(datagen.flow(img.reshape(1,101,101,3)))[0] / 255)
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Shear Range 0.2", y=0.93, fontsize=16)
#plt.tight_layout()
plt.show()


datagen = ImageDataGenerator(zoom_range = 0.2)
fig, axs = plt.subplots(2,2, figsize=(8,8))
ax = axs.flatten()
for a in ax:
    a.imshow(next(datagen.flow(img.reshape(1,101,101,3)))[0] / 255)
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Zoom Range 0.2", y=0.93, fontsize=16)
#plt.tight_layout()
plt.show()


datagen = ImageDataGenerator(horizontal_flip = True)
fig, axs = plt.subplots(2,2, figsize=(8,8))
ax = axs.flatten()
for a in ax:
    a.imshow(next(datagen.flow(img.reshape(1,101,101,3)))[0] / 255)
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Horizontal Flip", y=0.93, fontsize=16)
#plt.tight_layout()
plt.show()









