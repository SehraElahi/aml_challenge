import datetime
import keras
import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# TODO Find a way to turn off that red debugging spam from tensorflow, this does not work
from tensorflow.python.data import Dataset

tf.get_logger().setLevel('WARN')

print(f'Using GPU {tf.test.gpu_device_name()}')

# Global params and constants
WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.1
TRAIN_IMAGES_PATH = r'./dataset/train_set_labelled'
TEST_IMAGES_PATH = r'./dataset/test_set'
TRAIN_LABELS_PATH = r'./dataset/train_labels.csv'
PREDICTIONS_PATH = r'predictions.csv'
NUM_EXAMPLES = len(list(Path(TRAIN_IMAGES_PATH).rglob('*.jpg')))
NUM_CLASSES = len(list(Path(TRAIN_IMAGES_PATH).iterdir()))
print(f'Num classes: {NUM_CLASSES}  num samples: {NUM_EXAMPLES}')


def get_data_generators() -> tuple:
    # Generators allow to get the data in batches without having to worry about the memory
    generator = ImageDataGenerator(
        validation_split=VALIDATION_SPLIT,
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

    return train_gen, validation_gen, test_gen


def get_data_from_memory() -> tuple:
    train_xy = keras.preprocessing.image_dataset_from_directory(
        TRAIN_IMAGES_PATH,
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(WIDTH, HEIGHT),
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        seed=1,
        subset='training'
    )

    validation_xy = keras.preprocessing.image_dataset_from_directory(
        TRAIN_IMAGES_PATH,
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(WIDTH, HEIGHT),
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        seed=1,
        subset='validation'
    )

    test_x = keras.preprocessing.image_dataset_from_directory(
        TEST_IMAGES_PATH,
        batch_size=BATCH_SIZE,
        image_size=(WIDTH, HEIGHT),
        shuffle=False
    )

    return train_xy, validation_xy, test_x


def make_predictions_generator(model: keras.Model, test_gen: ImageDataGenerator) -> None:
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


def make_predictions_from_memory(model: keras.Model, test_set: Dataset) -> None:
    print('Making predictions for test set')
    predictions = model.predict(test_set, verbose=True, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1) + 1
    filenames = [f.name for f in Path(TEST_IMAGES_PATH, 'test_set').iterdir()]
    result = pd.DataFrame({'img_name': filenames, 'label': predictions})
    result = result.set_index('img_name')
    result.to_csv(f'predictions {datetime.datetime.now().strftime("%d-%m-%Y %Hh %Mm %Ss")}')


def get_callbacks() -> list:
    model_checkpoint = ModelCheckpoint(
        filepath=f'model {datetime.datetime.now().strftime("%d-%m-%Y %Hh %Mm %Ss")}',
        save_best_only=True,
        monitor='val_acc',
        verbose=1
    )
    return [model_checkpoint]
