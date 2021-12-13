from utils import *
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.applications import ResNet50

# from matplotlib import pyplot as plt

train_xy, validation_xy, test_x = get_data_from_memory()

# for _ in range(5):
#     img, label = train_gen.next()
#     # print(img[0].shape)
#     plt.imshow(img[0].astype(np.uint8))
#     plt.show()

# Initialize the Pretrained Model
feature_extractor = ResNet50(weights='imagenet',
                             input_shape=(WIDTH, HEIGHT, 3),
                             include_top=False)

num_layers = len(feature_extractor.layers)
for layer in feature_extractor.layers[:num_layers//2]:
    layer.trainable = False

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

model.fit(train_xy, epochs=EPOCHS, validation_data=validation_xy, callbacks=[model_checkpoint_callback])
