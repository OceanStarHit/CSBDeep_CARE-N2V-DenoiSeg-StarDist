
import os
# import sys
# import json
# import datetime
import numpy as np
import skimage.io
# from imgaug import augmenters as iaa

from glob import glob 

# import javabridge
# import bioformats

import imageio

train_data=\
'datasets/instance/training/NC1/train+\
datasets/instance/training/NC2/train+\
datasets/instance/training/T3/train+\
datasets/instance/training/T5/train+\
datasets/instance/training/T6/train+\
datasets/instance/training/T8/train'

val_data = \
'datasets/instance/training/NC1/val+\
datasets/instance/training/NC2/val+\
datasets/instance/training/T3/val+\
datasets/instance/training/T5/val+\
datasets/instance/training/T6/val+\
datasets/instance/training/T8/val'


def normalize_65535(im):
    im = im.astype(np.float32)
    min = np.min(im)
    max = np.max(im)
    im = (im-min)/(max-min)*65535
    return im

def nucleus_custom_dataset_preparation(train_data, custom_dataset_dir='datasets/nucleus_custom_embedseg_format/', subset='train'):
    print('Custom dataset directory: ', custom_dataset_dir)
    print('Subset: ', subset)

    custom_subset_imagdir = os.path.join(custom_dataset_dir, subset, 'images')
    if not os.path.exists(custom_subset_imagdir):
        os.mkdir(custom_subset_imagdir)
    custom_subset_maskdir = os.path.join(custom_dataset_dir, subset, 'masks')
    if not os.path.exists(custom_subset_maskdir):
        os.mkdir(custom_subset_maskdir)

    N_3D_images = 0
    N_2D_images = 0
    
    train_dirs = train_data.split('+')
    # val_dirs = val_data.split('+')
    print('Original data directory: ', train_dirs)

    train_imag_data_dirs = [os.path.join(dir, 'images') for dir in train_dirs]
    train_mask_data_dirs = [os.path.join(dir, 'labels') for dir in train_dirs]

    all_imags = [glob(os.path.join(imag_dir, '*.tif')) for imag_dir in train_imag_data_dirs]
    all_masks = [glob(os.path.join(mask_dir, '*.tif')) for mask_dir in train_mask_data_dirs]

    all_imags = sum(all_imags, [])
    all_imags.sort()
    all_masks = sum(all_masks, [])
    all_masks.sort()
    assert len(all_imags)==len(all_masks), 'error in the original dataset'

    print('3D images: ', len(all_imags))
    N_3D_images = len(all_imags)
    
    # javabridge.start_vm(class_path=bioformats.JARS)

    for imag, mask in zip(all_imags, all_masks):
        # imag_array = bioformats.ImageReader.read(imag)
        # mask_array = bioformats.ImageReader.read(mask)

        # sub_dir = 
        fname_split = imag.split('/')[-1]
        fname_split = imag.split('\\')[-1].split('.')
        fname_only = ''
        for i in range(len(fname_split)-1):
            fname_only = fname_only + fname_split[i]
        print(fname_only)

        imag_array = np.array(imageio.volread(imag))
        # imag_array = skimage.io.imread(imag)
        # print(imag_array.shape)

        mask_array = np.array(imageio.volread(mask))
        # imag_array = skimage.io.imread(mask)
        # print(mask_array.shape)

        N_slices = imag_array.shape[0]
        print('    Slices: ', N_slices)

        for n_slice in range(N_slices):
            
            fname_slice = fname_only + "_s" + str(n_slice)
            fname_slice_tif = fname_slice + '.tif'
            print('\t',fname_slice)
            
            imag_slice = normalize_65535(imag_array[n_slice]).astype(np.uint16)
            if len(imag_slice.shape)==3:
                imag_slice = imag_slice[..., 2]
            mask_slice = mask_array[n_slice].astype(np.uint16)

            skimage.io.imsave(os.path.join(custom_subset_imagdir, fname_slice_tif), imag_slice)
            skimage.io.imsave(os.path.join(custom_subset_maskdir, fname_slice_tif), mask_slice)

            # print()
            N_2D_images += 1

        print()

    # javabridge.kill_vm()

    print()
    
    return N_3D_images, N_2D_images



if __name__ == '__main__':
    custom_dataset_dir = 'datasets/nucleus_custom_embedseg_format/' # Root directory of the dataset
    if not os.path.exists(custom_dataset_dir):
        os.makedirs(custom_dataset_dir)
    print("Dataset: ", custom_dataset_dir)
    
    subset = 'train' # Dataset sub-directory
    print("Subset: ", subset)
    if not os.path.exists(os.path.join(custom_dataset_dir, subset)):
        os.mkdir(os.path.join(custom_dataset_dir, subset))
    N_3D_images_train, N_2D_images_train = nucleus_custom_dataset_preparation(
                                                                                train_data, 
                                                                                custom_dataset_dir, 
                                                                                subset=subset
                                                                            )

    subset = 'val' # Dataset sub-directory
    print("Subset: ", subset)
    if not os.path.exists(os.path.join(custom_dataset_dir, subset)):
        os.mkdir(os.path.join(custom_dataset_dir, subset))
    N_3D_images_val, N_2D_images_val = nucleus_custom_dataset_preparation(
                                                                            val_data, 
                                                                            custom_dataset_dir, 
                                                                            subset=subset
                                                                        )

    print('Train \t 3D images: ', N_3D_images_train, '\t 2D images: ', N_2D_images_train)                                                                    
    print('  Val \t 3D images: ',   N_3D_images_val, '\t 2D images: ',   N_2D_images_val)                                                                    
    
    print()