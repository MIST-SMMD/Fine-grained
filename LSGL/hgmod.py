''' Segmentation processing of images.
    Segmentation of the image using the Hugging face model with some necessary preprocessing methods.
'''

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import glob
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
import shutil

def Unipath(path):
    '''Convert erroneous paths into executable paths.
    Args:
        path(str): Raw processing path for input.
    Returns:
        path(str): Executable paths after processing.
    '''
    path = path.replace('\\','/')
    return path

def release_memory(segmodel):
    '''Free up video memory.
    Args:
        segmodel(str): User-selected image segmentation model.
    '''
    del class_queries_logits
    del feature_extractor
    del image
    del inputs
    del loader
    del masks_queries_logits
    del model
    del outputs
    if segmodel == 'maskformer-swin-large-ade' or segmodel == 'maskformer-swin-tiny-ade':
        del predicted_semantic_map
    if segmodel == 'maskformer-swin-large-coco':
        del predicted_panoptic_map
        del result
    print('Video Memory Released!')

def delete_mask(mask_images_path):
    '''Prevents mask files from participating in image segmentation.
    Args:
        mask_images_path(str): All image paths involved in image segmentation.
    Returns:
        paths(str): All image paths after masked image removal.
    '''
    paths = []
    for image in mask_images_path:
      if image.split("_")[-1].split(".")[0] != "seg":
        paths.append(image)
    return paths

def imagesegment(inppath,outimage,segmodel):
    '''Perform image segmentation.
    Args:
        inppath(str): Social Media Image Pathways for Engagement Segmentation.
        outimage(str): Output path for segmentation results.
        segmodel(str): User-selected segmentation model.
    Returns:
        paths(str): All image paths after masked image removal.
    '''
    mask_images_path_png = glob.glob(os.path.join(inppath+'/*.png'))
    mask_images_path_jpg = glob.glob(os.path.join(inppath+'/*.jpg'))
    mask_images_path = mask_images_path_png + mask_images_path_jpg
    mask_images_path = delete_mask(mask_images_path)
    # launch a progress bar
    scale = 150
    print(" Progress of image segmentation ".center(scale // 2,"-"))
    seg_bar = tqdm(total=len(mask_images_path))
    for inpimage in mask_images_path:
      
      # Open the image and compress it
      image = Image.open(inpimage)
      image = image.resize((640,480))
      
      # Processing data with selected models
      if segmodel == 'maskformer-swin-large-ade':
          predicted_semantic_map = maskformer_sla(image)
      elif segmodel == 'maskformer-swin-large-coco':
          predicted_semantic_map = maskformer_slc(image)
      elif segmodel == 'maskformer-swin-tiny-ade':
          predicted_semantic_map = maskformer_sta(image)

      # Use transforms functions from torchvision for the loader
      loader = transforms.Compose([
          transforms.ToTensor()])  

      unloader = transforms.ToPILImage()

      # Output PIL format image
      def tensor_to_PIL(tensor):
          image = tensor.cpu().clone()
          image = image.squeeze(0)
          image = unloader(image)
          return image
      predicted_semantic_map = torch.tensor(predicted_semantic_map, dtype=torch.float32)
      try:
          imgname = inpimage.split('/')
          tensor_to_PIL(predicted_semantic_map).save(outimage+'/'+imgname[-1])
      except:
          print('Please check if the output file path exists!')
      
      seg_bar.update(1)
      
      # Report completion
      print('Segmentation tasks completed!')
    seg_bar.close()

def maskformer_sla(image):
    '''Use the maskformer-swin-large-ade Model to process data.
    Args:
        image(str): Social Media Image Pathway for Engagement Segmentation.
    '''
    # Call the model
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-ade")
    inputs = feature_extractor(images=image, return_tensors="pt")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade")

    # Process the image
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    # Return the result
    return predicted_semantic_map

def maskformer_slc(image):
    '''Use the maskformer-swin-large-coco Model to process data.
    Args:
        image(str): Social Media Image Pathway for Engagement Segmentation.
    '''
    # Call the model
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Process the image
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result["segmentation"]
    
    # Return the result
    return predicted_panoptic_map

def maskformer_sta(image):
    '''Use the maskformer-swin-tiny-ade Model to process data.
    Args:
        image(str): Social Media Image Pathway for Engagement Segmentation.
    '''
    # Call the model
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-tiny-ade")
    inputs = feature_extractor(images=image, return_tensors="pt")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-tiny-ade")
    
    # Process the image
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    # Return the result
    return predicted_semantic_map

