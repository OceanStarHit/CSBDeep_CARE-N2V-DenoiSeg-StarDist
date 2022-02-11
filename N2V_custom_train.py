# %% [markdown]
# # Noise2Void - 2D Example for custom data

# %%
# We import all our dependencies.
from csbdeep.utils import _raise
from csbdeep.utils import plot_history
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

import tifffile
import numpy as np
from glob import glob
from os.path import join
from matplotlib import image
from matplotlib import pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# %% [markdown]
# # Training Data Preparation

class N2V_DataGenerator_custom(N2V_DataGenerator):
    def load_imgs(self, files, dims='YX'):
        """
        Helper to read a list of files. The images are not required to have same size,
        but have to be of same dimensionality.

        Parameters
        ----------
        files  : list(String)
                 List of paths to tiff-files.
        dims   : String, optional(default='YX')
                 Dimensions of the images to read. Known dimensions are: 'TZYXC'

        Returns
        -------
        images : list(array(float))
                 A list of the read tif-files. The images have dimensionality 'SZYXC' or 'SYXC'
        """
        assert 'Y' in dims and 'X' in dims, "'dims' has to contain 'X' and 'Y'."

        tmp_dims = dims
        for b in ['X', 'Y', 'Z', 'T', 'C']:
            assert tmp_dims.count(b) <= 1, "'dims' has to contain {} at most once.".format(b)
            tmp_dims = tmp_dims.replace(b, '')

        assert len(tmp_dims) == 0, "Unknown dimensions in 'dims'."

        if 'Z' in dims:
            net_axes = 'ZYXC'
        else:
            net_axes = 'YXC'

        move_axis_from = ()
        move_axis_to = ()
        for d, b in enumerate(dims):
            move_axis_from += tuple([d])
            if b == 'T':
                move_axis_to += tuple([0])
            elif b == 'C':
                move_axis_to += tuple([-1])
            elif b in 'XYZ':
                if 'T' in dims:
                    move_axis_to += tuple([net_axes.index(b)+1])
                else:
                    move_axis_to += tuple([net_axes.index(b)])

        imgs = []
        for f in files:
            if f.endswith('.tif') or f.endswith('.tiff'):
                imread = tifffile.imread
            elif f.endswith('.png'):
                imread = image.imread
            elif f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.JPG'):
                _raise(Exception("JPEG is not supported, because it is not loss-less and breaks the pixel-wise independence assumption."))
            else:
                _raise("Filetype '{}' is not supported.".format(f))

            img = imread(f).astype(np.float32)

            if len(img.shape)==3:
                for c in range(img.shape[2]):
                    img_1 = img[:,:,c]
                        # img = np.squeeze(img, axis=2)

                    # assert len(img.shape) == len(dims), "Number of image dimensions doesn't match 'dims'."

                    img_1 = np.moveaxis(img_1, move_axis_from, move_axis_to)

                    if not ('T' in dims):    
                        img_1 = img_1[np.newaxis]

                    if not ('C' in dims):
                        img_1 = img_1[..., np.newaxis]

                    imgs.append(img_1)

            elif len(img.shape)==2:
                img_1 = np.moveaxis(img, move_axis_from, move_axis_to)

                if not ('T' in dims):    
                    img_1 = img_1[np.newaxis]

                if not ('C' in dims):
                    img_1 = img_1[..., np.newaxis]

                imgs.append(img_1)

        return imgs


datagen = N2V_DataGenerator_custom()

# We load all the '.tif' files from the 'data' directory.
# The function will return a list of images (numpy arrays).
imgs_train = datagen.load_imgs_from_directory(directory = "datasets/nucleus_custom/train/**/images/", filter='*.tif', dims='YX')
imgs_valid = datagen.load_imgs_from_directory(directory = "datasets/nucleus_custom/val/**/images/", filter='*.tif', dims='YX')

# Let's look at the shape of the images.
print(imgs_train[0].shape, imgs_valid[0].shape)
# The function automatically added two extra dimensions to the images:
# One at the beginning, is used to hold a potential stack of images such as a movie.
# One at the end, represents channels.

# Lets' look at the images.
# We have to remove the added extra dimensions to display them as 2D images.
plt.imshow(imgs_train[2][0,...,0], cmap='magma')
plt.show()

plt.imshow(imgs_valid[2][0,...,0], cmap='magma')
plt.show()

# %%
# We will use the first image to extract training patches and store them in 'X'
patch_shape = (100,100)
# X = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape)
X = datagen.generate_patches_from_list(imgs_train, shape=patch_shape)

# We will use the second image to extract validation patches.
X_val = datagen.generate_patches_from_list(imgs_valid, shape=patch_shape)

# Patches are created so they do not overlap.
# (Note: this is not the case if you specify a number of patches. See the docstring for details!)
# Non-overlapping patches would also allow us to split them into a training and validation set 
# per image. This might be an interesting alternative to the split we performed above.

# Just in case you don't know how to access the docstring of a method:
# datagen.generate_patches_from_list

# Let's look at one of our training and validation patches.
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.imshow(X[0,...,0], cmap='magma')
plt.title('Training Patch');
plt.subplot(1,2,2)
plt.imshow(X_val[0,...,0], cmap='magma')
plt.title('Validation Patch');

# %% [markdown]
# # Configure

# train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch 
# is shown once per epoch. 
config = N2VConfig(X, unet_kern_size=3, 
                   train_steps_per_epoch=int(X.shape[0]/128), train_epochs=20, train_loss='mse', batch_norm=True, 
                   train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64), 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)

# Let's look at the parameters stored in the config-object.
vars(config)

# a name used to identify the model
model_name = 'n2v_custom_2D'
# the base directory in which our model will live
basedir = 'models_custom'

# We are now creating our network model.
# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.  
# model = N2V(config, model_name, basedir=basedir)

# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.  
model = N2V(config=None, name=model_name, basedir=basedir)

# model.config.train_epochs = 40

# %% [markdown]
# # Training

# We are ready to start training now.
history = model.train(X, X_val)

# %% [markdown]
# ### After training, lets plot training and validation loss.

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss']);

# %% [markdown]
# ## Export Model in BioImage ModelZoo Format
# See https://imagej.net/N2V#Prediction for details.
model.export_TF(name='Noise2Void - 2D SEM Example', 
                description='This is the 2D Noise2Void example trained on SEM data in python.', 
                authors=["Tim-Oliver Buchholz", "Alexander Krull", "Florian Jug"],
                test_img=X_val[0,...,0], axes='YX',
                patch_shape=patch_shape)

