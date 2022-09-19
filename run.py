# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:52:45 2022

@author: Navdeep Kumar
"""

__author__ = "Marganne Louis <louis.marganne@student.uliege.be>"
__contributors__ = ["Navdeep Kumar <nkumar@uliege.be>"]


from cytomine.models import ImageInstanceCollection, Job, AttachedFileCollection, Annotation, AnnotationCollection, JobData, TermCollection, Property, AbstractImage
from cytomine import CytomineJob

from tensorflow.keras.models import load_model

from shapely.geometry import Point

import tensorflow as tf
import numpy as np
import cv2 as cv

from utils import *

import joblib
import glob
import sys
import os


def find_by_attribute(att_fil, attr, val):
	return next(iter([i for i in att_fil if hasattr(i, attr) and getattr(i, attr) == val]), None)

def main(argv):
	with CytomineJob.from_cli(argv) as cj:
		cj.job.update(status=Job.RUNNING, progress=0, status_comment="Initialization of the prediction phase...")

		## 1. Create working directories on the machine:
		# - WORKING_PATH/images: store input images
		# - WORKING_PATH/rescaled: store rescaled version of images
		# - WORKING_PATH/in: store output from the model to use

		base_path = "{}".format(os.getenv("HOME"))
		working_path = os.path.join(base_path, str(cj.job.id))
		images_path = os.path.join(working_path, 'images/')
		rescaled_path = os.path.join(working_path, 'rescaled/')
		rescaled_images_path = os.path.join(rescaled_path, 'images/')
		in_path = os.path.join(working_path, 'in/')

		if not os.path.exists(working_path):
			os.makedirs(working_path)
			os.makedirs(images_path)
			os.makedirs(rescaled_path)
			os.makedirs(rescaled_images_path)
			os.makedirs(in_path)


		## 2. Parse input data
		# Select list of images corresponding to input
		images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		image_id_to_object = {image.id : image for image in images}
		if cj.parameters.images_to_predict == 'all':
			pred_images = images
		else:
			images_ids = [int(image_id) for image_id in cj.parameters.images_to_predict.split(',')]
			pred_images = [image_id_to_object[image_id] for image_id in images_ids]
		pred_images.sort(key=lambda x: x.id)

		# fetch terms
		terms = TermCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		term_ids = []
		for term in terms:
			id = term
			term_ids.append(id)
		

		# Load model depending upon the fish_type parameter
		if cj.parameters.fish_type=='zebrafish':
			model = load_model('/models/zebrafish_hrnet.hdf5')
		elif cj.parameters.fish_type=='medaka':
			model = load_model('/models/medaka_hrnet.hdf5')
		else:
			model = load_model('/models/seabream_hrnet.hdf5')
		

		## 3. Download the images
		cj.job.update(progress=5, statusComment='Downloading images...')

		for image in pred_images:
			image.download(dest_pattern=images_path+'%d.png' % image.id)


		## 4. Apply rescale to input
		cj.job.update(progress=50, statusComment='Rescaling images...')

		org_images = sorted(glob.glob(images_path+'*.png'))
		for i in range(len(org_images)):
			image = cv.imread(org_images[i], cv.IMREAD_UNCHANGED)
			im_name = os.path.basename(org_images[i])[:-4]
			re_img, _ = rescale_pad_img(image, None, 256)
			cv.imwrite(rescaled_images_path+im_name+'.png', re_img)


		## 5. Construct testing set with tensorflow and predict landmarks for each model
		cj.job.update(progress=80, statusComment='Predicting landmarks...')

		test_images = sorted(glob.glob(rescaled_images_path+'*.png'))
		batch_size = 2
		N = cj.parameters.n_landmarks

		pred_landmarks_list = []
		test_ds = tf.data.Dataset.from_tensor_slices((test_images, None))
		test_ds = test_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_ds = test_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_ds = test_ds.batch(batch_size)

			# Predict using the model
		preds = model.predict(test_ds)

			# Upscale prediction to original size
		pred_landmarks = []
		for j in range(len(test_images)):
			org_img = cv.imread(org_images[j])
			pred_mask = preds[j]
			pred_mask = np.reshape(pred_mask, newshape=(256,256, N))

			lm_list = []
			for k in range(N):
				x, y =  maskToKeypoints(pred_mask[:, :, k])
				lm_list.append((x, y))
				
			pred_lm = np.array(lm_list)
			up_size = org_img.shape[:2]
			up_lmks = up_lm(pred_lm, 256, up_size)
			pred_landmarks.append(up_lmks)

		pred_landmarks_list.append(pred_landmarks)


		## 6. Save landmarks as annotations in Cytomine
		annotation_collection = AnnotationCollection()
		for j, image in enumerate(pred_images):
			for k in range(N):
				lm = Point(pred_landmarks_list[i][j][k][0], image.height - pred_landmarks_list[i][j][k][1])
				annotation_collection.append(Annotation(location=lm.wkt, id_image=image.id, id_terms=term_ids[k].id, id_project=cj.parameters.cytomine_id_project))
		annotation_collection.save()

		cj.job.update(status=Job.TERMINATED, progress=100, statusComment='Job terminated.')


if __name__ == '__main__':
	main(sys.argv[1:])