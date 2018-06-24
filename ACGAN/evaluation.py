import os
import cv2
import glob
import numpy as np
import PIL
import torchvision
import pickle

def get_inception_images(nr_classes, attribute_list, save_dir):
    """we need about 50.000 images with uniform distribution over all classes"""
    iterations = int(50000/nr_classes) + 1
    for i in range(iterations):
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (nr_classes, opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for num in range(nr_classes)]) 
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)

        for j, image in enumerate(gen_imgs):
            # print(labels.cpu().numpy()[j])
            save_image(gen_imgs[j], save_dir + str(i) + '_' + attribute_list[j] + '.jpg')


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


# def SSIM():
	# TO DO
