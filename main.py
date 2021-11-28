import datetime
import keras
import pandas as pd
from pathlib import Path

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense

# TODO Find a way to turn off that red debugging spam from tensorflow, this does not work
tf.get_logger().setLevel('WARN')

print(f'Using GPU {tf.test.gpu_device_name()}')

# Global params and constants
WIDTH = 64
HEIGHT = 64
BATCH_SIZE = 64
EPOCHS = 1
TRAIN_IMAGES_PATH = r'./dataset/train_set_labelled'
TEST_IMAGES_PATH = r'./dataset/test_set'
TRAIN_LABELS_PATH = r'./dataset/train_labels.csv'
PREDICTIONS_PATH = r'predictions.csv'
NUM_EXAMPLES = len(list(Path(TRAIN_IMAGES_PATH).rglob('*.jpg')))
NUM_CLASSES = len(list(Path(TRAIN_IMAGES_PATH).iterdir()))
print(f'Num classes: {NUM_CLASSES}  num samples: {NUM_EXAMPLES}')

# Generators allow to get the data in batches without having to worry about the memory
generator = ImageDataGenerator(
    validation_split=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True
)
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


def get_model() -> keras.Model:
    """
    Build, compile and return the model
    """
    model = Sequential()
    model.add(Input(shape=(WIDTH, HEIGHT, 3)))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics='accuracy')
    return model


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


model = get_model()
model.summary()
model.fit(
    train_gen,
    validation_data=validation_gen,
    steps_per_epoch=10,
    validation_steps=1,
    epochs=EPOCHS,
    verbose=True,
)

make_predictions(model=model, test_gen=test_gen)
