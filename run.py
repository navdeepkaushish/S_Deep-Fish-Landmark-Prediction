# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:52:45 2022

@author: Navdeep Kumar
"""

__author__ = "Marganne Louis <louis.marganne@student.uliege.be>"
__contributors__ = ["Navdeep Kumar <nkumar@uliege.be>"]

from cytomine.models import ImageInstanceCollection, Job, AttachedFileCollection, Annotation, AnnotationCollection, \
	JobData, TermCollection, Property, AbstractImage
from cytomine import CytomineJob

from tensorflow.keras.models import load_model

from shapely.geometry import Point

import tensorflow as tf
import numpy as np
import cv2 as cv

from utils import *

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

		base_path = "{}".format(os.getenv("HOME"))
		working_path = os.path.join(base_path, str(cj.job.id))
		images_path = os.path.join(working_path, 'images/')

		if not os.path.exists(working_path):
			os.makedirs(working_path)
			os.makedirs(images_path)

		## 2. Parse input data
		# Select list of images corresponding to input
		images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		image_id_to_object = {image.id: image for image in images}
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
		if cj.parameters.fish_type == "seabream":
			model = load_model('d:/BioMedAqu/Projects/S_Deep-Fish-Landmark-Prediction/models/seabream_' + cj.parameters.model_to_use + '.hdf5')
		elif cj.parameters.fish_type == "medaka":
			model = load_model('/models/medaka_' + cj.parameters.model_to_use + '.hdf5')
		elif cj.parameters.fish_type == "zebrafish":
			model = load_model('/models/zebrafish_' + cj.parameters.model_to_use + '.hdf5')

		## 3. Download the images
		cj.job.update(progress=5, statusComment='Downloading images...')

		for image in pred_images:
			image.download(dest_pattern=images_path + '%d.png' % image.id)

		imgs_path = glob.glob(os.path.join(images_path,'*'))
		## 4. Apply rescale to input
		cj.job.update(progress=50, statusComment='Prepare images for execution...')
		model_size = cj.parameters.model_size
		N = cj.parameters.n_landmarks
		for i in range(len(imgs_path)):
			if cj.parameters.fish_type == 'seabream':
				img = cv.imread(imgs_path[i], 0) #grayscale
				filename = os.path.basename(imgs_path[i])
				fname, fext = os.path.splitext(filename)
				fname = int(fname)
				re_img, _ = rescale_pad_img(img, None, model_size )
				re_img = re_img.astype(np.float32)
				re_img = tf.cast(re_img, tf.float32) / 255.0
				re_img = tf.expand_dims(re_img, 0)
				re_img = tf.expand_dims(re_img, -1)


			else:
				img = cv.imread(imgs_path[i], cv.IMREAD_UNCHANGED)

				re_img, _ = rescale_pad_img(img, None, model_size) #rescaled and pad to model size (256 in our case)
				re_img = re_img.astype(np.float32)
				re_img = tf.cast(re_img, tf.float32) / 255.0 #noramalize
				re_img = tf.expand_dims(re_img, 0)
			pred_mask = model.predict(re_img)  # preicted heatmap mask
			pred_mask = np.reshape(pred_mask, newshape=(256, 256, N))

			lm_list = []
			for j in range(N):
				x, y = maskToKeypoints(pred_mask[:, :, j])
				lm_list.append((x, y))

			pred_lm = np.array(lm_list)
			up_size = img.shape[:2]
			up_lmks = up_lm(pred_lm, model_size, up_size)

			annotation_collection = AnnotationCollection()
			for k in range(N):
				lm = Point(up_lmks[k][0], img.shape[:1] - up_lmks[k][1])
				annotation_collection.append(
					Annotation(location=lm.wkt, id_image=fname, id_terms=term_ids[k].id, id_project=cj.parameters.cytomine_id_project))
			annotation_collection.save()

		cj.job.update(status=Job.TERMINATED, progress=100, statusComment='Job terminated.')


if __name__ == '__main__':
	main(sys.argv[1:])
