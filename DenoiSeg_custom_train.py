# %% [markdown]
# # DenoiSeg Train: custom data

# %%
# Here we are just importing some libraries which are needed to run this notebook.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt

import os
from glob import glob 
import imageio
from skimage.transform import resize

from denoiseg.models import DenoiSeg, DenoiSegConfig
from denoiseg.utils.misc_utils import shuffle_train_data, augment_data
from denoiseg.utils.seg_utils import *
from denoiseg.utils.compute_precision_threshold import measure_precision

from csbdeep.utils import plot_history

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
# Loading of the training images
train_images, train_masks = load_custom_dataset(dataset_dir, subset='train')
# Loading of the validation images
valid_images, valid_masks = load_custom_dataset(dataset_dir, subset='val')

print("Shape of train_images: {}".format(train_images.shape))
print("Shape of train_masks:  {}".format(train_masks.shape))
print("Shape of val_images:   {}".format(valid_images.shape))
print("Shape of val_masks:    {}".format(valid_masks.shape))

sample = 2
plt.figure(figsize=(10,5)); 
plt.subplot(1,2,1); plt.imshow(train_images[sample]); plt.axis('off'); plt.title('Train image');
plt.subplot(1,2,2); plt.imshow(train_masks[sample]); plt.axis('off'); plt.title('Train mask');
plt.show()
# %% [markdown]
# ## Small Amounts of Annotated Training Data
# With DenoiSeg we present a solution to train deep neural networks if only few annotated ground truth segmentations are available. We simulate such a scenary by zeroing out all but a fraction of the available training data. In the next cell you can specify the percentage of training images for which ground truth annotations are available.

# Set the number of annotated training images.
# Values: 0.0 (no annotated images) to total number of training images (all images have annotations)
number_of_annotated_training_images = 19
assert number_of_annotated_training_images >= 0.0 and number_of_annotated_training_images <=train_images.shape[0]
# Seed to shuffle training data (annotated GT and raw image pairs).
seed = 1 
# First we shuffle the training images to remove any bias.
X_shuffled, Y_shuffled = shuffle_train_data(train_images, train_masks, random_seed=seed)
# Here we convert the number of annotated images to be used for training as percentage of available training data.
percentage_of_annotated_training_images = float((number_of_annotated_training_images/train_images.shape[0])*100.0)
assert percentage_of_annotated_training_images >= 0.0 and percentage_of_annotated_training_images <=100.0
# Here we zero out the segmentations of those training images which are not part of the selected annotated images.
X_frac, Y_frac = zero_out_train_data(X_shuffled, Y_shuffled, fraction = percentage_of_annotated_training_images)


# Now we apply data augmentation to the training patches:
# Rotate four times by 90 degree and add flipped versions.
X, Y_train_masks = augment_data(X_frac, Y_frac)
X_val, Y_val_masks = valid_images, valid_masks


# Here we add the channel dimension to our input images.
# Dimensionality for training has to be 'SYXC' (Sample, Y-Dimension, X-Dimension, Channel)
X = X[...,np.newaxis]
Y = convert_to_oneHot(Y_train_masks)
X_val = X_val[...,np.newaxis]
Y_val = convert_to_oneHot(Y_val_masks)
print("Shape of X:     {}".format(X.shape))
print("Shape of Y:     {}".format(Y.shape))
print("Shape of X_val: {}".format(X_val.shape))
print("Shape of Y_val: {}".format(Y_val.shape))

# Next we look at a single sample. In the first column we show the input image, in the second column the background segmentation, in the third column the foreground segmentation and in the last column the border segmentation.
# With the parameter `sample` you can choose different training patches. You will notice that not all of them have a segmentation ground truth.
sample = 0
plt.figure(figsize=(20,5))
plt.subplot(1,4,1); plt.imshow(X[sample,...,0]); plt.axis('off'); plt.title('Raw validation image')
plt.subplot(1,4,2); plt.imshow(Y[sample,...,0], vmin=0, vmax=1, interpolation='nearest'); plt.axis('off'); plt.title('1-hot encoded background')
plt.subplot(1,4,3); plt.imshow(Y[sample,...,1], vmin=0, vmax=1, interpolation='nearest'); plt.axis('off'); plt.title('1-hot encoded foreground')
plt.subplot(1,4,4); plt.imshow(Y[sample,...,2], vmin=0, vmax=1, interpolation='nearest'); plt.axis('off'); plt.title('1-hot encoded border');
plt.show()

# %% [markdown]
# ### Configure network parameters
train_batch_size = 128
train_steps_per_epoch = min(400, max(int(X.shape[0]/train_batch_size), 10))

# You can choose how much relative importance (weight) to assign to denoising and segmentation tasks 
# by choosing appropriate value for `denoiseg_alpha` 
# (between `0` and `1`; with `0` being only segmentation and `1` being only denoising. Here we choose `denoiseg_alpha = 0.5`)
conf = DenoiSegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights = [1.0,1.0,5.0],
                      train_steps_per_epoch=train_steps_per_epoch, train_epochs=10, 
                      batch_norm=True, train_batch_size=train_batch_size, unet_n_first = 32, 
                      unet_n_depth=4, denoiseg_alpha=0.5, train_tensorboard=True)

vars(conf)

model_name = 'DenoiSeg_custom_2D'
basedir = 'models_custom'

# model = DenoiSeg(conf, model_name, basedir)

model = DenoiSeg(config=None, name=model_name, basedir=basedir)
model.config.train_epochs=50
# %%
history = model.train(X, Y, (X_val, Y_val))
history.history.keys()
plot_history(history, ['loss', 'val_loss'])

# ## Computing Threshold Value
# The network predicts 4 output channels:
# 1. The denoised input.
# 2. The background likelihoods.
# 3. The foreground likelihoods.
# 4. The border likelihoods.
# 
# We will threshold the foreground prediction image to obtain object segmentations. The optimal threshold is determined on the validation data. Additionally we can optimize the threshold for a given measure. In this case we choose the Average Precision (AP) measure.
threshold, val_score = model.optimize_thresholds(valid_images.astype(np.float32), valid_masks, measure=measure_precision())
print("The higest score of {} is achieved with threshold = {}.".format(np.round(val_score, 3), threshold))

# %%
print("Number of annotated images used for training:", number_of_annotated_training_images)
print("Considered alpha:", conf.denoiseg_alpha)

# Export your model for Fiji
model.export_TF(name='DenoiSeg - DSB2018 Example', 
                description='This is the 2D DenoiSeg example trained on DSB2018 data in python.', 
                authors=["Tim-Oliver Buchholz", "Mangal Prakash", "Alexander Krull", "Florian Jug"],
                test_img=X_val[0,...,0], axes='YX',
                patch_shape=(128, 128))
