import glob
import cv2
import numpy as np
import random
import numpy as np
from keras.utils import normalize

import model
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split


IMG_DIR_MASKS=r'D:\Segmentation_project\nucleus_images_blue\masks'
IMG_DIR_IMAGES= r'D:\Segmentation_project\nucleus_images_blue\images'

def read_images(directory):
    for img in glob.glob(directory+"/*"):
        image = cv2.imread(img)
        resized_img = cv2.resize(image/255.0  , (256, 256))

        yield resized_img
# read images and resize
masks_list =  np.array(list(read_images(IMG_DIR_MASKS)))
images_list = np.array(list(read_images(IMG_DIR_IMAGES)))


# rgb2gray
masks_list_gray = []
images_list_gray=[]
for i in images_list:
    gray = cv2.cvtColor(np.float32(i), cv2.COLOR_RGB2GRAY)
    images_list_gray.append(gray)

for i in masks_list:
    gray2 = cv2.cvtColor(np.float32(i), cv2.COLOR_RGB2GRAY)
    masks_list_gray.append(gray2)

# normalization
images_list_gray = np.expand_dims(normalize(np.array(images_list_gray), axis=1),3)
masks_list_gray = np.expand_dims((np.array(masks_list_gray)),3) /255.

#split data
X_train, X_test, y_train, y_test = train_test_split(images_list_gray, masks_list_gray, test_size = 0.10, random_state = 0)

#show random image with mask
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
# plt.subplot(122)
# plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
# plt.show()

IMG_HEIGHT = images_list_gray.shape[1]
IMG_WIDTH  = images_list_gray.shape[2]
IMG_CHANNELS = images_list_gray.shape[3]

def get_model():
    return model.simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

# history = model.fit(X_train, y_train,
#                     batch_size = 16,
#                     verbose=1,
#                     epochs=25,
#                     validation_data=(X_test, y_test),
#                     shuffle=False,
#                     callbacks=callbacks)

# np.save('my_history.npy',history.history)

# model.save('unet25_with_callbacks.h5')
model = load_model('unet25_with_callbacks.h5')

history=np.load('my_history.npy',allow_pickle='TRUE').item()
#Evaluate the model
#
#
	# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# test_img_other = cv2.imread('D:\Segmentation_project/image1_test.tif', 0)
# test_img_other = cv2.resize(test_img_other/255.0  , (256, 256))
# test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
# test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
# test_img_other_input=np.expand_dims(test_img_other_norm, 0)

# prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.title('External Image')
# plt.imshow(test_img_other, cmap='gray')
# plt.subplot(122)
# plt.title('Prediction of external Image')
# plt.imshow(prediction_other, cmap='gray')
# plt.show()
