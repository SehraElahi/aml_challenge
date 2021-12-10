from keras.layers import Layer
import cv2
import numpy as np
import tensorflow as tf


def image_preprocess(img):
    img = np.uint8(img)
    bordered = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # plt.imshow(cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB))

    # %%

    gray = cv2.cvtColor(bordered, cv2.COLOR_RGB2GRAY)
    # print(np.uint8(gray))
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(np.uint8(thresh), 0, 255), None)
    # plt.imshow(edges)
    # %%

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    edges = cv2.resize(edges, (224, 224))
    mask = np.zeros((224, 224), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    # plt.imshow(masked)
    # %%

    dst = cv2.bitwise_and(img, img, mask=mask)
    # segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # plt.imshow(segmented)
    # %%

    return dst


def image_tensor_func(img4d):
    results = []
    for img3d in img4d:
        rimg3d = image_preprocess(img3d)
        results.append(np.expand_dims(rimg3d, axis=0))
    return np.concatenate(results, axis=0)


class Segmentation(Layer):
    """Multiply inputs by `scale` and adds `offset`.
    For instance:
    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.
    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
    you would pass `scale=1./127.5, offset=-1`.
    The rescaling is applied both during training and inference.
    Input shape:
      Arbitrary.
    Output shape:
      Same as input.
    Arguments:
      scale: Float, the scale to apply to the inputs.
      offset: Float, the offset to apply to the inputs.
      name: A string, the name of the layer.
    """

    def __init__(self, name=None, **kwargs):
        super(Segmentation, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        xout = tf.py_function(image_tensor_func,
                              [inputs],
                              'float32',
                              name='cvOpt')
        # xout = K.stop_gradient( xout ) # explicitly set no grad
        # xout.set_shape( [xin.shape[0], 66, 200, xin.shape[-1]] ) # explicitly set output shape

        return xout

# seg = Segmentation()
