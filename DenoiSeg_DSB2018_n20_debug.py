# %% [markdown]
# # DenoiSeg Example: DSB 2018

# %%
# Here we are just importing some libraries which are needed to run this notebook.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from denoiseg.models import DenoiSeg, DenoiSegConfig
from denoiseg.utils.misc_utils import combine_train_test_data, shuffle_train_data, augment_data
from denoiseg.utils.seg_utils import *
from denoiseg.utils.compute_precision_threshold import measure_precision

from csbdeep.utils import plot_history

# %% [markdown]
# ## Downloaded Data Loading
# We created three versions of this dataset by adding Gaussian noise with zero mean and standard deviations 10 and 20. The dataset are marked with the suffixes n0, n10 and n20 accordingly.

# Choose the noise level you would like to look at:
# Values: 'n0', 'n10', 'n20'      
# Already 'n20' data downloaded
noise_level = 'n20'

# Loading of the training images
trainval_data =  np.load('./examples/examples_DenoiSeg/DenoiSeg_2D/data/DSB2018_{}/train/train_data.npz'.format(noise_level))
# train_images = trainval_data['X_train'].astype(np.float32)
train_images = trainval_data['X_train']
train_masks = trainval_data['Y_train']
# val_images = trainval_data['X_val'].astype(np.float32)
val_images = trainval_data['X_val']
val_masks = trainval_data['Y_val']

print("Shape of train_images: {}".format(train_images.shape))
print("Shape of train_masks:  {}".format(train_masks.shape))
print("Shape of val_images:   {}".format(val_images.shape))
print("Shape of val_masks:    {}".format(val_masks.shape))

sample = 0
plt.figure(figsize=(10,5))
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
X_val, Y_val_masks = val_images, val_masks

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

# %%
train_batch_size = 128
train_steps_per_epoch = min(400, max(int(X.shape[0]/train_batch_size), 10))

# %% [markdown]
# In the next cell, you can choose how much relative importance (weight) to assign to denoising 
# and segmentation tasks by choosing appropriate value for `denoiseg_alpha` (between `0` and `1`; with `0` being
# only segmentation and `1` being only denoising. Here we choose `denoiseg_alpha = 0.5`)

# %%
conf = DenoiSegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights = [1.0,1.0,5.0],
                      train_steps_per_epoch=train_steps_per_epoch, train_epochs=10, 
                      batch_norm=True, train_batch_size=train_batch_size, unet_n_first = 32, 
                      unet_n_depth=4, denoiseg_alpha=0.5, train_tensorboard=True)

vars(conf)

# %%
model_name = 'DenoiSeg_DSB18_n20'
basedir = 'models'
model = DenoiSeg(conf, model_name, basedir)

# %%
history = model.train(X, Y, (X_val, Y_val))

# %%
history.history.keys()

# %%
plot_history(history, ['loss', 'val_loss'])

# %% [markdown]
# ## Computing Threshold Value
# The network predicts 4 output channels:
# 1. The denoised input.
# 2. The background likelihoods.
# 3. The foreground likelihoods.
# 4. The border likelihoods.
# 
# We will threshold the foreground prediction image to obtain object segmentations. The optimal threshold is determined on the validation data. Additionally we can optimize the threshold for a given measure. In this case we choose the Average Precision (AP) measure.

# %%
threshold, val_score = model.optimize_thresholds(val_images.astype(np.float32), val_masks, measure=measure_precision())

print("The higest score of {} is achieved with threshold = {}.".format(np.round(val_score, 3), threshold))

# %% [markdown]
# ## Test Data
# Finally we load the test data and run the prediction.

# %%
test_data =  np.load('data/DSB2018_{}/test/test_data.npz'.format(noise_level), allow_pickle=True)
test_images = test_data['X_test']
test_masks = test_data['Y_test']

# %%
predicted_images, precision_result = model.predict_label_masks(test_images, test_masks, 0.5, 
                                                                   measure=measure_precision())
print("Average precision over all test images with threshold = {} is {}.".format(0.5, np.round(precision_result, 3)))

# %% [markdown]
# ### Visualize the results

# %%
sl = -10
fig = plt.figure()
plt.figure(figsize=(20,10))
plt.subplot(1, 3, 1); plt.imshow(test_images[sl]); plt.title("Raw image")
plt.subplot(1, 3, 2); plt.imshow(predicted_images[sl]); plt.title("Predicted segmentation")
plt.subplot(1, 3, 3); plt.imshow(test_masks[sl]); plt.title("Ground truth segmentation")
plt.show()

# %%
print("Number of annotated images used for training:", number_of_annotated_training_images)
print("Noise level:", noise_level)
print("Considered alpha:", conf.denoiseg_alpha)

# %% [markdown]
# Export your model for Fiji
model.export_TF(name='DenoiSeg - DSB2018 Example', 
                description='This is the 2D DenoiSeg example trained on DSB2018 data in python.', 
                authors=["Tim-Oliver Buchholz", "Mangal Prakash", "Alexander Krull", "Florian Jug"],
                test_img=X_val[0,...,0], axes='YX',
                patch_shape=(128, 128))
