''' Segmentation and masking.
    For segmenting images and masking non-building parts.
'''
from torchvision import transforms
from LSGL.hgmod import delete_mask
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import glob
import os
import numpy as np
import torch
import shutil

def getmask(maskpic):
  '''For reading image files.
    Args:
        maskpic(str): Path to the image file.
  '''
  return Image.open(maskpic)

def Unipath(userimage):
    '''Convert erroneous path into executable path.
    Args:
        userimage(str): Raw processing path for input.
    Returns:
        userimage(str): Executable paths after processing.
    '''
    userimage = userimage.replace('\\','/')
    return userimage

def Unipaths(paths):
    '''Convert erroneous paths into executable path.
    Args:
        paths(list): Raw processing path for input.
    Returns:
        paths(list): Executable paths after processing.
    '''
    for path in paths:
        path = path.replace('\\','/')
    return paths

def checkdoc(filepath):
    '''Check if the folder pointed to by the path exists.
    Args:
        filepath(str): Raw processing path for input.
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        
def imgscene(mask):
    '''Calculate the percentage of sky pixels in the image.
    Args:
        mask(str): Image path of segmentation results.
    '''
    mask_file_path = mask
    mask_images_path = glob.glob(os.path.join(mask_file_path + '*.png')) 

    for maskpic in  mask_images_path:
        mask = getmask(maskpic)
        mask = np.array(mask.convert('L'))
        sky = np.sum(mask[...,:] == 254)
        size = mask.size
        sky_percent = (sky / size) * 100
        print('Percentage of sky type like meta is {}%'.format(str(sky_percent)))

def proceed(mask, inp, segmodel):
    '''Converting segmentation result images into masks.
    Args:
        mask(str): Storage path for segmentation results.
        inp(str): Storage path for mask results.
        segmodel(str): User-selected segmentation model.
    '''
  
    # Check if the path exists
    checkdoc(mask)
    checkdoc(inp)
    
    # Retrieve all images under the path
    mask_file_path = mask
    inp_file_path = inp
    mask_images_path = glob.glob(os.path.join(mask_file_path+'*.png')) + glob.glob(os.path.join(mask_file_path+'*.jpg'))
    mask_images_path = delete_mask(mask_images_path)

    # launch progress bar
    scale = 150
    print(" Progress of masking ".center(scale // 2,"-"))
    mask_bar = tqdm(total=len(mask_images_path))

    # Iterate over all images
    for maskpic in  mask_images_path:
        # Convert Path Format
        maskpic = Unipath(maskpic)
        
        # Process the segmentation results to remove non-architectural image elements
        mask = getmask(maskpic)
        mask = np.array(mask.convert('L'))
        if segmodel == 'maskformer-swin-large-ade':
            mask = np.where(mask[..., :] == 255, 0, mask)
            mask = np.where(mask[..., :] == 250, 255, mask)
            mask = np.where(mask[..., :] == 252, 255, mask)
            mask = np.where(mask[..., :] == 239, 255, mask)
            mask = np.where(mask[..., :] == 235, 255, mask)
            mask = np.where(mask[..., :] == 236, 255, mask)
            mask = np.where(mask[..., :] == 176, 255, mask)
            mask = np.where(mask[..., :] == 244, 255, mask)
            mask = np.where(mask[..., :] == 254, 255, mask)
            mask = np.where(mask[..., :] == 255, 255, 0)
        else:
            mask = np.where((mask[...,:] > 243)&(mask[...,:] < 255), 0, 255)

        # Converts the data format of the mask matrix to uint8
        mask = mask.astype(np.uint8)
        Image.fromarray(mask)

        # Converting matrix data to images
        img = Image.fromarray(mask)

        # Setting the output path
        temp = maskpic.split("/")[-1].split(".")
        img.save(inp + "/" + temp[0] + "_seg.jpg")

        # Update progress bar
        mask_bar.update(1)
    mask_bar.close()

        
    # Returns on mission completion
    print('Mask processing task completed!')

