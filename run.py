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