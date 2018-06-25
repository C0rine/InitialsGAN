import os
import cv2
import glob
import numpy as np
import PIL
import torchvision
import pickle
import random
import sys
from scipy import signal
from scipy import ndimage
from skimage.measure import compare_ssim as ssim

def nearest_neighbor(nr_neighbors, nr_gen_img_start, nr_gen_img_end, gen_img_dir, save_dir):
	print('Starting with Nearest Neighbor search for ' + str(nr_neighbors) + ' neighbors.')
	
	# get the images as lists of numpy arrays
	gen_imgs_list = []
	gen_imgs = glob.glob(gen_img_dir + '*.jpg')
	for c, gen_img in enumerate(gen_imgs[nr_gen_img_start:nr_gen_img_end]):
		image = cv2.imread(gen_img)
		gen_imgs_list.append(image)

	# get real imgs as list of numpy arrays (pre-pickled)
	with open('real_img.pkl', 'rb') as f:
		real_imgs_list = pickle.load(f)

	# check if save directory exists, and if not, create one
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)

	# go through each generated image and find the nearest neighbor(s)
	for i, gen_img in enumerate(gen_imgs_list):
		print('Looking at img ' + str(i))
		# create a save list for the NN image names and distances
		neighbors_idx = []
		neighbors = []
		distances = []

		# get the distance to all images from the real dataset
		for j, real_img in enumerate(real_imgs_list):
			# get distance between the two images
			current_distance = np.sum((gen_img - real_img)**2)
			# add it to the list
			distances.append(current_distance)

		# horizontally stack the images to create a list of images
		hor_stack = gen_img

		# Get the nr_neighbors smallest distance indices
		for k in range(nr_neighbors):
			index = distances.index(min(distances))
			neighbors_idx.append(index)
			neighbors.append(real_imgs_list[index])
			# add the neighbors to the image stack
			hor_stack = np.hstack((hor_stack, neighbors[k]))

			del distances[index]

		print('Neighbors: ')
		print(neighbors_idx)

		# save the image
		im = PIL.Image.fromarray(hor_stack)
		im.save(save_dir + 'nn_' + str(i) + '.jpg')

	print('Nearest Neighbor search completed and images saved to: ' + save_dir)


def SSIM_classmeans(gen_imgs_dir, class_list, idx1=-5, idx2=-4):

	all_ssims = []
	path = gen_imgs_dir

	for clas in class_list:

		print('SSIM for class: ' + clas)
		# take random 100 pairs of imgs from gen_imgs_dir
		hundred_ssims = []

		# create a list with all images from the folder that below to the class
		class_imgs = []
		for img in os.listdir(gen_imgs_dir):
			idx1 = img[:idx2].rfind('_')+1
			if img[idx1:idx2] == clas:
				class_imgs.append(img)
		if(len(class_imgs) < 2):
			# too small no samples, add 0 to fill
			all_ssims.append(1)
			break

		for i in range(100):
			# get a random pair of iamages from the list of images for one class
			random_img1, random_img2 = random.sample(class_imgs, 2)
			# convert them to numpy arrays
			random_np_img1 = np.asarray(PIL.Image.open(gen_imgs_dir + random_img1))
			random_np_img2 = np.asarray(PIL.Image.open(gen_imgs_dir + random_img2))

			# calculate SSIM between the imgs and add to list
			hundred_ssims.append(ssim(random_np_img1, random_np_img2, multichannel=True))

		# return the mean of the list
		all_ssims.append(np.mean(hundred_ssims))
		print(np.mean(hundred_ssims))

	# return a list of the SSIMs 
	return all_ssims