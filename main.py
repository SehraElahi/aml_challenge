import datetime
import keras
import pandas as pd
from pathlib import Path
import keras_tuner as kt

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import adam_v2
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout

# TODO Find a way to turn off that red debugging spam from tensorflow, this does not work
tf.get_logger().setLevel('WARN')

print(f'Using GPU {tf.test.gpu_device_name()}')

# Global params and constants
WIDTH = 64
HEIGHT = 64
BATCH_SIZE = 64
EPOCHS = 10
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
    featurewise_std_normalization=True,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.4, 1.5]
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


def get_model(hp) -> keras.Model:
    """
    Build, compile and return the model
    """

    units_dense = hp.Int('units_dense', 16, 64, 16)
    # rate_dropout = hp.Float('rate_dropout', 0.2, 0.5, 0.1)
    num_conv_layers = hp.Int('num_conv_layers', 2, 4)
    dense_dropout_rate = hp.Float('dense_dropout_rate', 0.2, 0.5, 0.1)

    model = Sequential()
    model.add(Input(shape=(WIDTH, HEIGHT, 3)))
    for i in range(num_conv_layers):
        filters_conv = hp.Int(f'filters_conv{i}', min_value=8, max_value=64, step=8)
        # should_add_dropout = hp.Boolean(f'dropout{i}')

        model.add(Conv2D(filters=filters_conv, kernel_size=3, activation='relu', padding='same'))
        # if should_add_dropout:
        #     model.add(Dropout(rate=rate_dropout))
        model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Conv2D(filters=filters_conv, kernel_size=3, activation='relu', padding='same'))
        # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(units=units_dense, activation='relu'))
    model.add(Dropout(dense_dropout_rate))
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


# model = get_model(kt.HyperParameters())
# model.summary()
# model.fit(
#     train_gen,
#     validation_data=validation_gen,
#     # steps_per_epoch=10,
#     # validation_steps=1,
#     epochs=EPOCHS,
#     verbose=True,
# )

tuner = kt.RandomSearch(
    hypermodel=get_model,
    objective="accuracy",
    max_trials=3,
    executions_per_trial=1,
    overwrite=True,
    directory="keras_tuner_search",
    project_name="aml_challenge",
)

tuner.search_space_summary()

tuner.search(train_gen, epochs=1)

# Get the top 2 models.
models = tuner.get_best_models(num_models=2)

best_model = models[0]
# Build the model.
# # Needed for `Sequential` without specified `input_shape`.
# best_model.build(input_shape=(None, 28, 28))
best_model.summary()
tuner.results_summary()

best_hps = tuner.get_best_hyperparameters()
print('Best hyperparameters:' + str(best_hps[0].values))

model = get_model(best_hps[0])
model.summary()
model.fit(
    train_gen,
    validation_data=validation_gen,
    # steps_per_epoch=10,
    # validation_steps=1,
    epochs=EPOCHS,
    verbose=True,
)

make_predictions(model=model, test_gen=test_gen)
