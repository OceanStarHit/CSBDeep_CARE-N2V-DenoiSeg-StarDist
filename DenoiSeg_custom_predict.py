# %% [markdown]
# # DenoiSeg Prediction: custom data

# %%
# Here we are just importing some libraries which are needed to run this notebook.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

import os
from glob import glob 
import imageio
from skimage.transform import resize

from denoiseg.models import DenoiSeg
from denoiseg.utils.compute_precision_threshold import measure_precision

# %% [markdown]
# ## Dataset Loading
def load_custom_dataset(dataset_dir='datasets/nucleus_custom_embedseg_format/', subset='train'):
    print('Dataset directory: ', dataset_dir)
    print('           Subset: ', subset)

    train_images = []
    train_masks = []

    imag_dir = os.path.join(dataset_dir, subset, 'images')
    imag_files = glob(os.path.join(imag_dir, '*.tif'))
    imag_files.sort()

    mask_dir = os.path.join(dataset_dir, subset, 'masks')
    mask_files = glob(os.path.join(mask_dir, '*.tif'))
    mask_files.sort()
    
    for imag_file, mask_file in zip(imag_files, mask_files):
        image = np.array(imageio.volread(imag_file)).astype(np.float32)
        image = resize(image, (256, 256), order = 0)
        train_images.append(image)

        mask = np.array(imageio.volread(mask_file)).astype(np.float32)
        mask = resize(mask, (256, 256), order = 0)
        train_masks.append(mask)
        
    train_images = np.array(train_images).astype(np.float32)
    train_masks = np.array(train_masks).astype(np.uint16)

    return train_images, train_masks


dataset_dir = './datasets/nucleus_custom_embedseg_format'
valid_images, valid_masks = load_custom_dataset(dataset_dir, subset='val')
print("Shape of val_images:   {}".format(valid_images.shape))
print("Shape of val_masks:    {}".format(valid_masks.shape))

# %%
model_name = 'DenoiSeg_custom_2D'
basedir = 'models_custom'

model = DenoiSeg(config=None, name=model_name, basedir=basedir)
model.config.denoiseg_alpha=0.2
# %% [markdown]
# ## Predict
predicted_images, precision_result = model.predict_label_masks(valid_images, valid_masks, 0.5, 
                                                                   measure=measure_precision())
print("Average precision over all test images with threshold = {} is {}.".format(0.5, np.round(precision_result, 3)))

# ### Visualize the results
sl = 0
fig = plt.figure(); plt.figure(figsize=(20,10)); 
plt.subplot(1, 3, 1); plt.imshow(valid_images[sl]); plt.title("Raw image")
plt.subplot(1, 3, 2); plt.imshow(predicted_images[sl]); plt.title("Predicted segmentation"); 
plt.subplot(1, 3, 3); plt.imshow(valid_masks[sl]); plt.title("Ground truth segmentation")
plt.show()

# %%
print("Considered alpha:", model.config.denoiseg_alpha)

