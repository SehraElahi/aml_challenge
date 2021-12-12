from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from PIL import Image
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from custom_layers import Segmentation
from matplotlib import pyplot as plt

# Global params and constants
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 64
EPOCHS = 50
TRAIN_IMAGES_PATH = r'./dataset/train_set_labelled'
TEST_IMAGES_PATH = r'./dataset/test_set'
TRAIN_LABELS_PATH = r'./dataset/train_labels.csv'
PREDICTIONS_PATH = r'predictions.csv'
NUM_EXAMPLES = len(list(Path(TRAIN_IMAGES_PATH).rglob('*.jpg')))
NUM_CLASSES = len(list(Path(TRAIN_IMAGES_PATH).iterdir()))
print(f'Num classes: {NUM_CLASSES}  num samples: {NUM_EXAMPLES}')


def make_predictions(model: keras.Model, test_gen: ImageDataGenerator):
    """
    Output a CSV with model's predictions on test set that is ready to be submitted to Kaggle.
    The file will be created in the main directory of the project, named 'predictions <current_time>'
    """
    predictions = model.predict(test_gen, verbose=True, batch_size=BATCH_SIZE)
    # Get names of test files in the same order they were used for predictions
    file_names = list(map(lambda x: x.split('\\')[1], test_gen.filenames))
    # Obtain final labels for predictions, add one since classes start from one
    predictions = predictions.argmax(axis=1) + 1
    result = pd.DataFrame({'img_name': file_names, 'label': predictions})
    result = result.set_index('img_name')
    # Save the CSV file to main project directory
    result.to_csv(f'predictions {datetime.datetime.now().strftime("%d-%m-%Y %Hh %Mm %Ss")}')


# Generators allow to get the data in batches without having to worry about the memory
generator = ImageDataGenerator(
    validation_split=0.2,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    # # width_shift_range=0.2,
    # # height_shift_range=0.2,
    # # rescale=1. // 255,
    # # shear_range=0.2,
    # zoom_range=0.2,
    # # horizontal_flip=True,
    # # vertical_flip=True,
    fill_mode='nearest',
    # brightness_range=[0.8, 1.5]
    # preprocessing_function=image_preprocess   # commented this because segmentation happens inside NN
)
# sample_image_paths = [str(path) for path in list(Path(TRAIN_IMAGES_PATH).rglob('*.jpg'))[:1000]]
# # print([str(path) for path in image_paths[:5]])
# x = [np.array(Image.open(fname)) for fname in sample_image_paths]
# print(x)
# generator.fit(x)

train_gen = generator.flow_from_directory(
    directory=TRAIN_IMAGES_PATH,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(WIDTH, HEIGHT),
    shuffle=True,
    subset='training'
)
validation_gen = generator.flow_from_directory(
    directory=TRAIN_IMAGES_PATH,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(WIDTH, HEIGHT),
    shuffle=True,
    subset='validation'
)
test_gen = generator.flow_from_directory(
    directory=TEST_IMAGES_PATH,
    class_mode=None,
    batch_size=BATCH_SIZE,
    target_size=(WIDTH, HEIGHT),
    shuffle=False
)

for _ in range(5):
    img, label = train_gen.next()
    # print(img[0].shape)
    plt.imshow(img[0].astype(np.uint8))
    plt.show()

# Initialize the Pretrained Model
feature_extractor = ResNet50(weights='imagenet',
                             input_shape=(WIDTH, HEIGHT, 3),
                             include_top=False)

model = Sequential()
model.add(tf.keras.Input(shape=(WIDTH, HEIGHT, 3)))
# model.add(Segmentation())
model.add(feature_extractor)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

# Compile it
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print The Summary of The Model
model.summary()

checkpoint_filepath = 'training\\weights.{epoch:02d}-{val_loss:.2f}.hdf5'
Path(checkpoint_filepath.split('\\')[0]).mkdir(parents=True, exist_ok=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(train_gen, epochs=EPOCHS, validation_data=validation_gen, callbacks=[model_checkpoint_callback])
