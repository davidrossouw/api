import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam


def create(input_shape, num_classes):
	"""
	CNN Keras model with 6 convolutions.
	:param input_shape: input shape, generally X_train.shape[1:]
	:return: Keras model, RMS prop optimizer
	"""
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(256, (3, 3), padding='same')) 
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model


class SimpsonClassifier(object):

	def __init__(self, weights_path, pic_size=64):

		self.weights_path = weights_path
		self.input_shape = (pic_size, pic_size, 3)
		self.map_characters = {0: 'abraham_grampa_simpson', 
						  1: 'apu_nahasapeemapetilon',
						  2: 'bart_simpson', 
						  3: 'charles_montgomery_burns', 
						  4: 'chief_wiggum', 
						  5: 'comic_book_guy', 
						  6: 'edna_krabappel', 
						  7: 'homer_simpson', 
						  8: 'kent_brockman', 
						  9: 'krusty_the_clown', 
						  10: 'lisa_simpson', 
						  11: 'marge_simpson', 
						  12: 'milhouse_van_houten', 
						  13: 'moe_szyslak', 
						  14: 'ned_flanders', 
						  15: 'nelson_muntz', 
						  16: 'principal_skinner', 
						  17: 'sideshow_bob'}

		self.num_classes = len(self.map_characters)

		# Load model
		self.model = create(self.input_shape, self.num_classes)
		self.model.load_weights(weights_path)


	def run(self, img):
		'''
		Run the model on the input image. Return the prediction and probability
		in a dictionary
		'''
		y_pred = self.model.predict_classes(img)[0]
		y_pred_name = self.map_characters[y_pred]
		y_prob = round(self.model.predict_proba(img)[0][y_pred], 2)
		return {'y_pred': y_pred_name, 'y_prob': y_prob}


def load_img(path, size=64):
	'''
	Load image from file, resize and return numpy array
	'''
	img = cv2.imread(path)
	img = cv2.resize(img, (size, size)).astype('float32') / 255.
	img = np.expand_dims(img, axis=0)
	return img


# def run_model(img):
# 	'''
# 	Run the model on the input image. Return the prediction and probability
# 	in a dictionary
# 	'''
# 	y_pred = model.predict_classes(img)[0]
# 	y_pred_name = map_characters[y_pred]
# 	y_prob = round(model.predict_proba(img)[0][y_pred], 2)
# 	return {'y_pred': y_pred_name, 'y_prob': y_prob}

img = load_img('/Users/davidrossouw/Documents/my_projects/api/fastapi/app/data/principal_skinner_3.jpg', 64)
model = SimpsonClassifier(weights_path='/Users/davidrossouw/Documents/my_projects/api/fastapi/app/data/weights.best.hdf5', pic_size=64)
print (model.run(img))


# img = load_img()
# print(run_model(img))
