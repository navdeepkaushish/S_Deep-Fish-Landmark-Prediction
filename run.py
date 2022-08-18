# -*- coding: utf-8 -*-
<<<<<<< HEAD
"""
Created on Wed Aug 17 10:52:45 2022

@author: Navdeep Kumar
"""

from __future__ import print_function

import glob
import os
import sys
import json
from shapely.geometry import Point, Polygon

from PIL import Image
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow.keras.backend as K
from utils import *
from loss_functions import *

import argparse
import json
import logging

from cytomine import Cytomine
from cytomine import CytomineJob
from cytomine.models import (
	Property,
	Annotation,
	AnnotationTerm,
	AnnotationCollection,
	Project,
	ImageInstanceCollection,
	Job)
#==============================================================================
def main(argv):
	with CytomineJob.from_cli(argv) as conn:
		conn.job.update(status=Job.RUNNING, progress=0, statusComment='Intialization...')
		base_path = "{}".format(os.getenv('HOME'))  # Mandatory for Singularity
		working_path = os.path.join(base_path, str(conn.job.id))


		# Select images to process
		images = ImageInstanceCollection().fetch_with_filter('project', conn.parameters.cytomine_id_project)
		if conn.parameters.cytomine_id_images != 'all':  # select only given image instances = [image for image in image_instances if image.id in id_list]
			images = [_ for _ in images if _.id
					  in map(lambda x: int(x.strip()),
							 conn.parameters.cytomine_id_images.split(','))]
		#images_id = [image.id for image in images]

		# Download selected images into 'working_directory'
		img_path = os.path.join(working_path, 'images')
		# if not os.path.exists(img_path):
		os.makedirs(img_path)

		for image in conn.monitor(
				images, start=2, end=50, period=0.1,
				prefix='Downloading images into working directory...'):
			fname, fext = os.path.splitext(image.filename)
			if image.download(dest_pattern=os.path.join(
					img_path,
					"{}{}".format(image.id, fext))) is not True:  # images are downloaded with image_ids as names
				print('Failed to download image {}'.format(image.filename))

		#========= loading model according to fish type =======================

		if conn.parameters.fish =='zebrafish':
			with tf.device('/cpu:0'):
				model = load_model('/models/zebra_hrnet.hdf5', compile=True)
		elif conn.parameters.fish == 'medaka':
			with tf.device('/cpu:0'):
				model = load_model('/models/medaka_hrnet.hdf5', compile=True)
		else:
			with tf.device('/cpu:0'):
				model = load_model('/models/seabream_hrnet.hdf5', compile=True)

		 # Prepare image file paths from image directory for execution
		conn.job.update(progress=50,
						statusComment="Preparing data for execution..")
		image_paths = glob.glob(os.path.join(img_path, '*'))

		model_size = conn.paremeters.model_size #256x256 in our case
		N = conn.parameters.n_landmarks

		for i in range(len(img_path)):

			org_img = cv.imread(img_path[i], cv.IMREAD_UNCHANGED)
			filename = os.path.basename(image_paths[i])[:-4]
			#fname = int(fname)

			re_img, re_lm = rescale_pad_img(org_img, None, model_size)

			tf_img = array_to_img(re_img)
			tf_img = tf.cast(tf_img, tf.float32) / 255.0  # normalize the image
			tf_img = tf.expand_dims(tf_img, 0)

			pred_mask = model.predict(tf_img)
			pred_mask = np.reshape(pred_mask, newshape=(model_size,model_size, N))

			lm_list = []
			for j in range(N):
				x,y =  maskToKeypoints(pred_mask[:, :, j])
				lm_list.append((x,y))

			pred_lm = np.array(lm_list)
			up_size = org_img.shape[:2]
			lm = up_lm(pred_lm, model_size, up_size)
			image_id = next((x.id for x in images if x.originalFilename == filename), None)

			lm1 = Point(lm[0])
			lm2 = Point(lm[1])
			lm3 = Point(lm[2])
			lm4 = Point(lm[3])
			lm5 = Point(lm[4])
			lm6 = Point(lm[5])
			lm7 = Point(lm[6])
			lm8 = Point(lm[7])
			lm9 = Point(lm[8])
			lm10 = Point(lm[9])
			lm11 = Point(lm[10])
			lm12 = Point(lm[11])
			lm13 = Point(lm[12])
			lm14 = Point(lm[13])
			lm15 = Point(lm[14])
			lm16 = Point(lm[15])
			lm17 = Point(lm[16])
			lm18 = Point(lm[17])
			lm19 = Point(lm[18])
			lm20 = Point(lm[19])
			lm21 = Point(lm[20])
			lm22 = Point(lm[21])
			lm23 = Point(lm[22])
			lm24 = Point(lm[23])
			lm25 = Point(lm[24])

			annotations = AnnotationCollection()
			annotations.append(Annotation(location=lm1.wkt, id_image=image_id, id_terms = 549311142, id_project=549311125))
			annotations.append(Annotation(location=lm2.wkt, id_image=image_id, id_terms = 549311148, id_project=549311125))
			annotations.append(Annotation(location=lm3.wkt, id_image=image_id, id_terms = 549311156, id_project=549311125))
			annotations.append(Annotation(location=lm4.wkt, id_image=image_id, id_terms = 549311162, id_project=549311125))
			annotations.append(Annotation(location=lm5.wkt, id_image=image_id, id_terms = 549311170, id_project=549311125))
			annotations.append(Annotation(location=lm6.wkt, id_image=image_id, id_terms=549311176, id_project=549311125))
			annotations.append(Annotation(location=lm7.wkt, id_image=image_id, id_terms=549311184, id_project=549311125))
			annotations.append(Annotation(location=lm8.wkt, id_image=image_id, id_terms=549311190, id_project=549311125))
			annotations.append(Annotation(location=lm9.wkt, id_image=image_id, id_terms=549311198, id_project=549311125))
			annotations.append(Annotation(location=lm10.wkt, id_image=image_id, id_terms=549311204, id_project=549311125))
			annotations.append(Annotation(location=lm11.wkt, id_image=image_id, id_terms=549311212, id_project=549311125))
			annotations.append(Annotation(location=lm12.wkt, id_image=image_id, id_terms=549311218, id_project=549311125))
			annotations.append(Annotation(location=lm13.wkt, id_image=image_id, id_terms=549311226, id_project=549311125))
			annotations.append(Annotation(location=lm14.wkt, id_image=image_id, id_terms=549311232, id_project=549311125))
			annotations.append(Annotation(location=lm15.wkt, id_image=image_id, id_terms=549311240, id_project=549311125))
			annotations.append(Annotation(location=lm16.wkt, id_image=image_id, id_terms=549311246, id_project=549311125))
			annotations.append(Annotation(location=lm17.wkt, id_image=image_id, id_terms=549311254, id_project=549311125))
			annotations.append(Annotation(location=lm18.wkt, id_image=image_id, id_terms=549311262, id_project=549311125))
			annotations.append(Annotation(location=lm19.wkt, id_image=image_id, id_terms=549311270, id_project=549311125))
			annotations.append(Annotation(location=lm20.wkt, id_image=image_id, id_terms=549311276, id_project=549311125))
			annotations.append(Annotation(location=lm21.wkt, id_image=image_id, id_terms=549311282, id_project=549311125))
			annotations.append(Annotation(location=lm22.wkt, id_image=image_id, id_terms=549311290, id_project=549311125))
			annotations.append(Annotation(location=lm23.wkt, id_image=image_id, id_terms=549311296, id_project=549311125))
			annotations.append(Annotation(location=lm24.wkt, id_image=image_id, id_terms=549311304, id_project=549311125))
			annotations.append(Annotation(location=lm25.wkt, id_image=image_id, id_terms=549311310, id_project=549311125))
			annotations.save()

		conn.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)  # 524787186

if __name__ == '__main__':
	main(sys.argv[1:])





=======

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
		

		# Fetch data from the trained model(s)
		models = []
		parameters_hash = []

		jobs_ids = [int(job_id) for job_id in cj.parameters.model_to_use.split(',')]
		for job_id in jobs_ids:
			tr_model_job = Job().fetch(job_id)
			attached_files = AttachedFileCollection(tr_model_job).fetch()
			tr_model = find_by_attribute(attached_files, 'filename', '%d_model.hdf5' % job_id)
			tr_model_filepath = in_path + '%d_model.hdf5' % job_id
			tr_model.download(tr_model_filepath, override=True)
			tr_parameters = find_by_attribute(attached_files, 'filename', '%d_parameters.joblib' % job_id)
			tr_parameters_filepath = in_path + '%d_parameters.joblib' % job_id
			tr_parameters.download(tr_parameters_filepath, override=True)

			# Load fetched data
			models.append(load_model(tr_model_filepath))
			parameters_hash.append(joblib.load(tr_parameters_filepath))


		## 3. Download the images
		cj.job.update(progress=5, statusComment='Downloading images...')

		for image in pred_images:
			image.download(dest_pattern=images_path+'%d.tif' % image.id)


		## 4. Apply rescale to input
		cj.job.update(progress=50, statusComment='Rescaling images...')

		org_images = sorted(glob.glob(images_path+'*.tif'))
		for i in range(len(org_images)):
			image = cv.imread(org_images[i], cv.IMREAD_UNCHANGED)
			im_name = os.path.basename(org_images[i])[:-4]
			re_img, _ = rescale_pad_img(image, None, 256)
			cv.imwrite(rescaled_images_path+im_name+'.png', re_img)


		## 5. Construct testing set with tensorflow and predict landmarks for each model
		cj.job.update(progress=80, statusComment='Predicting landmarks...')

		test_images = sorted(glob.glob(rescaled_images_path+'*.png'))

		pred_landmarks_list = []
		for i in range(len(jobs_ids)):
			test_ds = tf.data.Dataset.from_tensor_slices((test_images, None))
			test_ds = test_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
			test_ds = test_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
			test_ds = test_ds.batch(parameters_hash[i]['model_batch_size'])

			# Predict using the model
			preds = models[i].predict(test_ds)

			# Upscale prediction to original size
			pred_landmarks = []
			for j in range(len(test_images)):
				org_img = cv.imread(org_images[j])
				pred_mask = preds[j]
				pred_mask = np.reshape(pred_mask, newshape=(256,256, parameters_hash[i]['N']))

				lm_list = []
				for k in range(parameters_hash[i]['N']):
					x, y =  maskToKeypoints(pred_mask[:, :, k])
					lm_list.append((x, y))
				
				pred_lm = np.array(lm_list)
				up_size = org_img.shape[:2]
				up_lmks = up_lm(pred_lm, 256, up_size)
				pred_landmarks.append(up_lmks)

			pred_landmarks_list.append(pred_landmarks)


		## 6. Save landmarks as annotations in Cytomine
		annotation_collection = AnnotationCollection()
		for i in range(len(jobs_ids)):
			for j, image in enumerate(pred_images):
				for k in range(parameters_hash[i]['N']):
					lm = Point(pred_landmarks_list[i][j][k][0], image.height - pred_landmarks_list[i][j][k][1])
					annotation_collection.append(Annotation(location=lm.wkt, id_image=image.id, id_terms=[parameters_hash[i]['cytomine_id_terms'][k]], id_project=cj.parameters.cytomine_id_project))
		annotation_collection.save()


		## 7. Save prediction in a TPS file
		# Construct dictionnaries mapping image/term name to image/name id
		terms = TermCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		terms_names = {term.id : term.name for term in terms}

		with open(f'{cj.job.id}.TPS', 'w') as f:
			# Compute total number of landmarks
			LM = 0
			for i in range(len(jobs_ids)):
				LM += parameters_hash[i]['N']

			for i, landmarks in enumerate(zip(*pred_landmarks_list)):
				# Get info about current image
				image_name = pred_images[i].filename
				image_height = pred_images[i].height

				# Fetch scale
				try:
					prop = Property(pred_images[i]).fetch(key='scale')
					image_scale = prop.value
				except AttributeError:
					try:
						abstract_image = AbstractImage().fetch(pred_images[i].baseImage)
						prop = Property(abstract_image).fetch(key='scale')
						image_scale = prop.value
					except AttributeError:
						image_scale = 'x.xxxxxx'

				# Sort landmarks in accordance to their name
				lm = np.concatenate(landmarks)
				lm_ids = [parameters_hash[j]['cytomine_id_terms'] for j in range(len(jobs_ids))]
				lm_ids = np.concatenate(lm_ids)
				lm_names = [terms_names[lm_id] for lm_id in lm_ids]
				lm = lm[np.argsort(lm_names)]

				# Write number of landmarks
				f.write(f'LM={LM}\n')

				# Write landmarks
				for landmark in lm:
					f.write("%.5f %.5f\n" % (landmark[0], image_height - landmark[1]))

				# Write image name and scale
				f.write('IMAGE='+image_name+'\n')
				f.write('SCALE='+image_scale+'\n')


		# Upload TPS file to Cytomine
		job_data = JobData(id_job=cj.job.id, key='Prediction TPS file', filename=f'{cj.job.id}.TPS')
		job_data = job_data.save()
		job_data.upload(f'{cj.job.id}.TPS')

		cj.job.update(status=Job.TERMINATED, progress=100, statusComment='Job terminated.')


if __name__ == '__main__':
	main(sys.argv[1:])
>>>>>>> 24bfdbb97aee6fa1bfe33df0847843090d1e351c
