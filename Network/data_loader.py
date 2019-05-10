import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
#from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image as sklearn_image

class ImageFolder(data.Dataset):
	def __init__(self, root, image_size = 32, mode = 'train', augmentation_prob = 0, PPorLS = 'LS'):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
        
		# If using Lung-segmented scans as train/validation/test sets.
		if PPorLS == 'LS':
			self.GT_paths = root[:-1]+'_GT/'
		# If using Preprocessed scans as train/validation/test sets.
		elif PPorLS == 'PP':
			self.GT_paths = root[:-4]+'_GT/'
		else:
			print('Must include the source of train/validation/test sets. PPorLS has to be PP or LS.')

		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		self.PPorLS = PPorLS
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]

		
		if self.PPorLS == 'LS':
			filename = image_path.split('/')[-1][:64]
			location_and_extension = image_path.split('/')[-1][75:]

		else:
			filename = image_path.split('/')[-1][:64]
			location_and_extension = image_path.split('/')[-1][78:]

		GT_path = self.GT_paths + filename + '_GT_'+ location_and_extension

		# somehow the imread function pads the image to 4 channels. For the grayscale image we only need the first channel.
		image = plt.imread(image_path)[:,:,0]
		GT = plt.imread(GT_path)[:,:,0]

		# pad the images to 32 x 32
		target_img_size = 32
		if np.shape(GT)[0] > target_img_size:
			print('Found an image larger than 32. New size is {}'.format(np.shape(GT)[0]))
		pad_size_topleft = int((target_img_size - np.shape(GT)[0])/2)
		pad_size_bottomright = int(target_img_size - pad_size_topleft - np.shape(GT)[0])
		image = np.pad(image, ((pad_size_topleft, pad_size_bottomright), (pad_size_topleft, pad_size_bottomright)), 'edge')
		GT = np.pad(GT, ((pad_size_topleft, pad_size_bottomright), (pad_size_topleft, pad_size_bottomright)), 'edge')
        
		# Crop off the image if it exceeds 32 x 32
		image = image[0:32, 0:32]
		GT = GT[0:32, 0:32]
        
		return image, GT

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers = 1, mode = 'train', augmentation_prob = 0, shuffle = True, PPorLS = 'LS'):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size = image_size, mode = mode,augmentation_prob = augmentation_prob, PPorLS = PPorLS)
	data_loader = data.DataLoader(dataset = dataset,
								  batch_size = batch_size,
								  shuffle = shuffle,
								  num_workers = num_workers)
	return data_loader

def prediction_loader(image_path, batch_size = 1, num_workers = 1):
	"""The loader to help perform the whole slice prediction."""
	
	# somehow the imread function pads the image to 4 channels. For the grayscale image we only need the first channel.
	whole_image = plt.imread(image_path)[:,:,0]
	image_size = np.shape(whole_image)

	# Currently, I am not interested in writing code to cover the unlikely case where the image size is smaller than 32 x 32. So...
	if (image_size[0] < 32) or (image_size[1] < 32):
		print('The image size is smaller than 32 x 32')

	# Chop up the image into patches of 32 x 32
	patch_size = (32, 32)
	patches = sklearn_image.extract_patches_2d(whole_image, patch_size)
	if (np.shape(patches)[1] != 32) or (np.shape(patches)[2] != 32):
		print('Shape of the patches are not as expected. Shape is: {}'.format(np.shape(patches)))

	# Put that to data.DataLoader and return the result
	data_loader = data.DataLoader(dataset = patches, batch_size = batch_size, shuffle = False, num_workers = num_workers)
	
	return data_loader