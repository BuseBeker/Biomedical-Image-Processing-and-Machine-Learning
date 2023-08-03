# 181805057 - Kardelen Gel
# 181805067 - Buse Latife Beker

# Required libraries are imported.
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2DTranspose
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

#Of the 10 images in the dataset, we reserve 9 for the Unet model for training and 1 for testing.
folder_path = "images/Train_images/"
file_extension = ".tif"

train_images = []
for i in range(1, 10):
    image_path = folder_path + "image_" + str(i) + file_extension
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    train_images.append(image)
        
test_image = cv2.imread('images/Train_images/image_10.tif')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  
     
folder_path = "images/Train_masks/"

train_masks = []
for i in range(1, 10):
    image_path = folder_path + "mask_" + str(i) + file_extension
    mask_image = cv2.imread(image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)  
    train_masks.append(mask_image)
    
test_mask  = cv2.imread('images/Train_masks/mask_10.tif')
test_mask  = cv2.cvtColor(test_mask , cv2.COLOR_BGR2GRAY) 

# We make the necessary dimensions to use the images in the Unet model. 
image_height = 256
image_width = 256
channels = 1

train_images = np.stack(train_images)
train_masks = np.stack(train_masks)

train_images = train_images.reshape((9, image_height, image_width, channels))
train_masks = train_masks.reshape((9, image_height, image_width, 1))
test_image_unet = test_image.reshape((1, image_height, image_width, channels))
test_mask_unet = test_mask.reshape((1, image_height, image_width, 1))

#We define the Unet model.
def unet(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu')(inputs)
    conv11 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu')(conv1)
    bn1 = BatchNormalization(axis=3)(conv11)
    bn1 = Activation("relu")(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu')(pool1)
    conv22 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu')(conv2)
    bn2 = BatchNormalization(axis=3)(conv22)
    bn2 = Activation("relu")(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu')(pool2)
    conv33 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu')(conv3)
    bn3 = BatchNormalization(axis=3)(conv33)
    bn3 = Activation("relu")(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation='relu')(pool3)
    conv44 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation='relu')(conv4)
    bn4 = BatchNormalization(axis=3)(conv44)
    bn4 = Activation("relu")(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation='relu')(pool4)
    conv55 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation='relu')(conv5)
    bn5 = BatchNormalization(axis=3)(conv55)
    bn5 = Activation("relu")(bn5)

    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn5), conv4], axis=3)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation='relu')(up6)
    conv66 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation='relu')(conv6)
    bn6 = BatchNormalization(axis=3)(conv66)
    bn6 = Activation("relu")(bn6)

    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn6), conv3], axis=3)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu')(up7)
    conv77 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu')(conv7)
    bn7 = BatchNormalization(axis=3)(conv77)
    bn7 = Activation("relu")(bn7)

    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn7), conv2], axis=3)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu')(up8)
    conv88 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu')(conv8)
    bn8 = BatchNormalization(axis=3)(conv88)
    bn8 = Activation("relu")(bn8)

    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn8), conv1], axis=3)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu')(up9)
    conv99 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu')(conv9)
    bn9 = BatchNormalization(axis=3)(conv99)
    bn9 = Activation("relu")(bn9)

    conv10 = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(bn9)

    return Model(inputs=[inputs], outputs=[conv10])

# We train the Unet model. We calculate the training time information using the time function.
start_time = time.time()

input_shape = (256, 256, 1)
model = unet(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_masks, epochs=30, batch_size=1)

end_time = time.time()
training_time = end_time - start_time

# We make predictions on the test data and print the scores requested from us.
predictions = model.predict(test_image_unet)

loss, accuracy = model.evaluate(test_image_unet, test_mask_unet)

threshold = 0.7
binary_predictions = (predictions > threshold).astype(np.uint8)
intersection = np.logical_and(binary_predictions, test_mask_unet)
union = np.logical_or(binary_predictions, test_mask_unet)
iou = np.sum(intersection) / np.sum(union)

print("Accuracy Score:", accuracy)
print("IoU Score:", iou)
print("Training Time:", training_time)

# We saved the Unet model and the predicted image.
import pickle

filename = "sandstone_model_unet"
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(test_image_unet)

segmented = result.reshape((test_image.shape))

from matplotlib import pyplot as plt
plt.subplot(221)
plt.imshow(test_image)
plt.subplot(222)
plt.imshow(test_mask, cmap ='jet')
plt.subplot(224)
plt.imshow(segmented, cmap ='jet')
plt.imsave('segmented_test_image_unet.jpg', segmented, cmap ='jet')


# Of the 10 images in the dataset, we reserve 9 for the Random Forest model for training and 1 for testing.
folder_path = "images/Train_images/"
train_images2 = []
for i in range(1, 10):
    image_path = folder_path + "image_" + str(i) + file_extension
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = image.reshape(-1)
    train_images2.append(image)

test_image_rf = test_image.reshape(-1) 

folder_path = "images/Train_masks/"
train_masks2 = []
for i in range(1, 10):
    image_path = folder_path + "mask_" + str(i) + file_extension
    mask_image = cv2.imread(image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)  
    mask_image = mask_image.reshape(-1)
    mask_image = (mask_image > 0).astype(int)
    train_masks2.append(mask_image)
    
test_mask_rf = test_mask.reshape(-1)

train_images_rf = np.array(train_images2)
train_masks_rf = np.array(train_masks2)

# We define the Random Forest model.
# We train the Random Forest model. We calculate the training time information using the time function.
start_time = time.time()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_images_rf, train_masks_rf)

end_time = time.time()
training_time = end_time - start_time

# We make predictions on the test data and print the scores requested from us.
test_image_prediction = rf_model.predict([test_image_rf])

test_mask_rf = test_mask_rf.reshape(test_image.shape)
test_image_prediction = test_image_prediction.reshape(test_image.shape)

accuracy = accuracy_score(test_mask_rf.flatten(), test_image_prediction.flatten())

intersection = np.logical_and(test_mask_rf, test_image_prediction)
union = np.logical_or(test_mask_rf, test_image_prediction)
iou_score = np.sum(intersection) / np.sum(union)

print("Accuracy:", accuracy)
print("IoU Score:", iou_score)
print("Training Time:", training_time)

# We saved the Random Forest model and the predicted image.
import pickle

filename = "sandstone_rf_model"
pickle.dump(rf_model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([test_image_rf])
result = result.reshape((1, -1))

segmented = result.reshape(test_image.shape)

plt.subplot(221)
plt.imshow(test_image)
plt.subplot(222)
plt.imshow(test_mask, cmap ='jet')
plt.subplot(224)
plt.imshow(segmented, cmap ='jet')
plt.imsave('segmented_test_image_rf.jpg', segmented, cmap ='jet')

