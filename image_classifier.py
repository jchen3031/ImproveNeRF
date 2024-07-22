import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

base_model = ResNet50(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

number_classes = len(glob.glob("imageclassifer/train/*"))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(number_classes, activation='softmax')(x)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    'imageclassifer/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    'imageclassifer/validation',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 200
steps_per_epoch = 8
validation_steps = 1
train = False # change to train or test mode
test_dir = './data/nerf_llff_data/leaves/images'


def load_and_predict_images(test_dir, model):
    for img_path in glob.glob(test_dir + '/*.jpg'):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        print(f'Image: {img_path} - Predicted Class: {predicted_class[0]}')


if __name__ == '__main__':
    if train:
        model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps
        )
        model.save('classifer_model.h5')
    else:
        model.load('classifer_model.h5')
        load_and_predict_images(test_dir, model)
