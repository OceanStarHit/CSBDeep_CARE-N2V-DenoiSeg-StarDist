# %% [markdown]
# # Noise2Void - 2D Example for custom data
# 
# __Note:__ This notebook expects a trained model and will only work if you have executed the `01_training.ipynb` beforehand.

# %%
# We import all our dependencies.
from csbdeep.utils import _raise
from csbdeep.utils import plot_history
from csbdeep.io import save_tiff_imagej_compatible
from n2v.models import N2V

import skimage.io
from tqdm import tqdm
import tifffile
import numpy as np
from glob import glob
from os.path import join
from matplotlib import image
from matplotlib import pyplot as plt

# %% [markdown]
# ## Load the Network

# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.  
model_name = 'n2v_custom_2D'
basedir = 'models_custom'
model = N2V(config=None, name=model_name, basedir=basedir)

# # In case you do not want to load the weights that lead to lowest validation loss during 
# # training but the latest computed weights, you can execute the following line:
# model.load_weights('weights_last.h5')

# %% [markdown]
# ## Prediction

def normalize_65535(im):
    im = im.astype(np.float32)
    min = np.min(im)
    max = np.max(im)
    im = (im-min)/(max-min)*65535
    return im

def predict_replace_files(img_files):
    
    # for f in tqdm(img_files):
    for f in img_files:
        imag = tifffile.imread(f).astype(np.float32)

        if len(imag.shape)==2:
            pred = model.predict(imag, axes='YX')
            # save_tiff_imagej_compatible(f, pred, axes='YX')
        elif len(imag.shape)==3:
            pred = np.zeros_like(imag)
            for i in range(imag.shape[2]):
                pred[:,:,i] = model.predict(imag[:,:,i], axes='YX')
            # save_tiff_imagej_compatible(f, pred, axes='YXC')
        print(f, imag.shape, pred.shape)
        pred = normalize_65535(pred)
        skimage.io.imsave(f, pred.astype(np.uint16))
        
def predict_place_files(img_files):
    
    # for f in tqdm(img_files):
    for f in img_files:
        imag = tifffile.imread(f).astype(np.float32)

        if len(imag.shape)==2:
            pred = model.predict(imag, axes='YX')
            # save_tiff_imagej_compatible(f, pred, axes='YX')
        elif len(imag.shape)==3:
            pred = np.zeros_like(imag)
            for i in range(imag.shape[2]):
                pred[:,:,i] = model.predict(imag[:,:,i], axes='YX')
            # save_tiff_imagej_compatible(f, pred, axes='YXC')

        f2 = f.split('.')[-2] + '_n2v.tif'
        print(f2, imag.shape, pred.shape)
        pred = normalize_65535(pred)
        skimage.io.imsave(f2, pred.astype(np.uint16))
        

directory = "datasets/nucleus_custom_for_n2v_test/train/images/"
filter='*.tif'
img_files = glob(join(directory, filter))
# files.sort()
# predict_replace_files(img_files)
predict_place_files(img_files)

directory = "datasets/nucleus_custom_for_n2v_test/val/images/"
filter='*.tif'
img_files = glob(join(directory, filter))
# files.sort()
# predict_replace_files(img_files)
predict_place_files(img_files)
            
# %% [markdown]
# ### Show results on training data...

# %%
# # Let's look at the results.
# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# plt.imshow(input_train[:1500:,:1500],cmap="magma")
# plt.title('Input');
# plt.subplot(1,2,2)
# plt.imshow(pred_train[:1500,:1500],cmap="magma")
# plt.title('Prediction');

# # %% [markdown]
# # ### Show results on validation data...

# # %%
# # Let's look at the results.
# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# plt.imshow(input_val,cmap="magma")
# plt.title('Input');
# plt.subplot(1,2,2)
# plt.imshow(pred_val,cmap="magma")
# plt.title('Prediction');

# # %% [markdown]
# # ## Save Results

# # %%
# save_tiff_imagej_compatible('pred_train.tif', pred_train, axes='YX')
# save_tiff_imagej_compatible('pred_validation.tif', pred_val, axes='YX')


