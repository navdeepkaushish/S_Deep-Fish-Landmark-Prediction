# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

__author__ = "Navdeep Kumar <nkumar@uliege.be>"
__contributors__ = ["Marganne Louis <louis.marganne@student.uliege.be>"]


from tensorflow.keras.preprocessing.image import img_to_array

import albumentations as A
import tensorflow as tf
import numpy as np
import cv2 as cv

import os

# Custom data augmentation
transforms = A.Compose([
					   A.RandomBrightnessContrast(),
					   A.Affine(scale=(0.8, 1), 
								translate_px= (0, 10), 
								rotate= (-10, 10), 
								shear=(0, 10), 
								interpolation=0, # nearest
								mode=2, # mode replicate
								fit_output=False), 
					   A.HorizontalFlip(p=0.3),
					   ], 
						keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))


# Read image and convert to tensorflow object
def parse_data(image, lm):
	image_content = tf.io.read_file(image)
	image = tf.io.decode_png(image_content, channels=3)
	image = tf.image.resize(image, (256, 256))
	return  image, lm

# Normalize image
def normalize(image, lm):
	image = tf.cast(image, tf.float32) / 255.0
	return image, lm

# Apply custom data augmentation
def aug_fn(image, lm):
	image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
	lm = lm.numpy()
	aug_data = transforms(image=image, keypoints=lm)
	image_aug = aug_data['image'] 
	lm_aug = aug_data['keypoints']
	lm_aug = np.array(lm_aug, dtype=np.int32)
	return image_aug, lm_aug

# Apply custom data augmentation through py_function
def aug_apply(image, lm, N):
	image, lm = tf.py_function(aug_fn, (image, lm), (tf.float32, tf.float32))
	image.set_shape((256,256,3))
	lm.set_shape((N,2))
	return image, lm

# Exponential probability function which describes the spread of the heatmap
def _exp(xL, yL, sigma, H, W):
	xx, yy = np.mgrid[0:H:1, 0:W:1]
	kernel = np.exp(-np.log(2) * 0.5 * (np.abs(yy - xL) + np.abs(xx - yL)) / sigma)
	kernel = np.float32(kernel)
	return kernel

# Gaussian probibility function which describes the spread of the heatmap
def _gaussian(xL, yL, sigma, H, W):
	xx, yy = np.mgrid[0:H:1, 0:W:1]
	kernel = np.exp(-0.5 * (np.square(yy - xL) + np.square(xx - yL)) / np.square(sigma))
	kernel = np.float32(kernel)
	return kernel


# Convert an image to an heatmap
def _convertToHM(img, keypoints, sigma, prob_function):
	H = img.shape[0]
	W = img.shape[1]
	nKeypoints = len(keypoints)

	img_hm = np.zeros(shape=(H, W, nKeypoints), dtype=np.float32)

	for i in range(0, nKeypoints):
		x = keypoints[i][0]
		y = keypoints[i][1]

		channel_hm = _exp(x, y, sigma, H, W) if prob_function == 'exp' else _gaussian(x, y, sigma, H, W)
		img_hm[:, :, i] = channel_hm
	
	img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints, 1))
	return img, img_hm

# Convert an image to a heatmap through py_function
def to_hm(image, lm, N, sigma, prob_function):
	image, lm = tf.py_function(_convertToHM, [image, lm, sigma, prob_function], [tf.float32, tf.float32])
	image.set_shape((256,256,3))
	lm.set_shape((256*256*N, 1))
	return image, lm

# Rescale images and landmarks to the desired size
def rescale_pad_img(image, landmarks, desired_size):
	
	h, w = image.shape[:2]
	
	aspect = w / h
	
	if aspect > 1 : #horizontal image
		new_w = desired_size
		new_h = int(desired_size * h / w)
		offset = int(new_w - new_h)
		if offset %  2 != 0: #odd offset
			top = offset // 2 + 1
			bottom = offset // 2
		else:
			top = bottom = offset // 2
		
		dim = (new_w, new_h)
		re_img = cv.resize(image, dim, interpolation = cv.INTER_AREA)
		pad_img = cv.copyMakeBorder(re_img, top, bottom, 0, 0, cv.BORDER_REPLICATE)

		if landmarks is not None:
			x = landmarks[:,0]
			y = landmarks[:,1]
			new_x = x * new_w / w
			new_x = new_x.astype(int)
			new_y = y * new_h / h + offset // 2
			new_y = new_y.astype(int)
			pad_lm = np.vstack((new_x, new_y)).T
		else:
			pad_lm = None
			
	elif aspect < 1:  #vertical image
		new_h = desired_size
		new_w = int(desired_size * w / h)
		offset = int(np.ceil((new_h - new_w) // 2))
		if offset %  2 != 0: #odd offset
			top = offset - 1
			bottom = offset
		else:
			top = bottom = offset
		dim = (new_w, new_h)
		re_img = cv.resize(image, dim, interpolation = cv.INTER_AREA)
		pad_img = cv.copyMakeBorder(re_img, top, bottom, 0, 0, cv.BORDER_CONSTANT, value=0)

		if landmarks is not None:
			new_x = x * new_w / w + offset //2
			new_x = new_x.astype(int)
			new_y = y * new_h / h
			new_y = new_y.astype(int)
			pad_lm = np.vstack((new_x, new_y)).T
		else:
			pad_lm = None
	
	return pad_img, pad_lm

# Get coordinates of a predicted landmark based on its heatmap
def maskToKeypoints(mask):
	kp = np.unravel_index(np.argmax(mask, axis=None), shape=(256,256))
	return kp[1], kp[0]

# Upscale landmarks to upsize
def up_lm(lmks, curr_size, upsize):
	asp_ratio = upsize[0]/upsize[1] # h/w
	
	w = curr_size
	h = w * asp_ratio
	
	up_w = upsize[1]
	up_h = upsize[0]
	
	offset = w - h
	x_lm = lmks[:,0]
	y_lm = lmks[:,1]
	
	y_lm = y_lm - offset//2
	
	up_y_lm = y_lm * up_h / h
	up_x_lm = x_lm * up_w / w
	
	return np.vstack((up_x_lm, up_y_lm)).T