
#!/usr/bin/env python3

import argparse
import numpy as np
from keras.layers import Convolution2D, Cropping2D, GlobalAveragePooling2D, Multiply, Add, Conv2DTranspose
from keras.layers.advanced_activations import ReLU
from keras import Input, Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def create_model():
    input1 = Input(shape=(200, 200, 3))
    conv1 = Convolution2D(32, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution1', input_shape=(200, 200, 3))(input1)
    conv2 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution2')(conv1)
    conv3 = Convolution2D(64, 2, 2, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution3')(conv2)
    conv4 = Convolution2D(128, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution4')(conv3)
    conv5 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution5')(conv4)

    pooling1 = GlobalAveragePooling2D(keepdims=True)(conv5)
    conv6 = Convolution2D(8, 1, 1, activation="relu", padding="valid", name='Convolution6')(pooling1)
    conv7 = Convolution2D(64, 1, 1, activation="sigmoid", padding="valid", name='Convolution7')(conv6)
    scale1 = Multiply()([conv5, conv7])
    deconv1 = Conv2DTranspose(64, 2, 2, activation=ReLU(negative_slope=0.2), padding="valid", name="Deconvolution1")(scale1)

    crop1 = Cropping2D(4)(conv2)

    eltwise1 = Add()([deconv1, crop1])

    conv8 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution8')(eltwise1)
    deconv2 = Conv2DTranspose(3, 4, 2, padding="same", name="Deconvolution2")(conv8)

    conv9 = Convolution2D(32, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution9')(deconv2)
    conv10 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution10')(conv9)

    conv11 = Convolution2D(64, 2, 2, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution11')(conv10)
    conv12 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution12')(conv11)
    conv13 = Convolution2D(128, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution13')(conv12)

    pooling2 = GlobalAveragePooling2D(keepdims=True)(conv13)
    conv14 = Convolution2D(16, 1, 1, activation="relu", padding="valid", name='Convolution14')(pooling2)
    conv15 = Convolution2D(128, 1, 1, activation="sigmoid", padding="valid", name='Convolution15')(conv14)
    scale2 = Multiply()([conv13, conv15])

    conv16 = Convolution2D(128, 2, 2, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution16')(scale2)
    conv17 = Convolution2D(256, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution17')(conv16)
    conv18 = Convolution2D(128, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution18')(conv17)

    pooling3 = GlobalAveragePooling2D(keepdims=True)(conv18)
    conv19 = Convolution2D(16, 1, 1, activation="relu", padding="valid", name='Convolution19')(pooling3)
    conv20 = Convolution2D(128, 1, 1, activation="sigmoid", padding="valid", name='Convolution20')(conv19)
    scale3 = Multiply()([conv18, conv20])
    deconv3 = Conv2DTranspose(128, 2, 2, activation=ReLU(negative_slope=0.2), padding="valid", name="Deconvolution3")(scale3)

    crop2 = Cropping2D(4)(scale2)

    eltwise2 = Add()([deconv3, crop2])

    conv21 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution21')(eltwise2)
    conv22 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution22')(conv21)

    pooling4 = GlobalAveragePooling2D(keepdims=True)(conv22)
    conv23 = Convolution2D(8, 1, 1, activation="relu", padding="valid", name='Convolution23')(pooling4)
    conv24 = Convolution2D(64, 1, 1, activation="sigmoid", padding="valid", name='Convolution24')(conv23)
    scale4 = Multiply()([conv22, conv24])

    deconv4 = Conv2DTranspose(64, 2, 2, activation=ReLU(negative_slope=0.2), padding="valid", name="Deconvolution4")(scale4)

    crop3 = Cropping2D(16)(conv10)

    eltwise3 = Add()([deconv4, crop3])

    conv25 = Convolution2D(64, 3, 1, activation=ReLU(negative_slope=0.2), padding="valid", name='Convolution25')(eltwise3)
    conv26 = Convolution2D(3, 3, 1, padding="valid", name='Convolution26')(conv25)

    crop4 = Cropping2D(20)(deconv2)

    eltwise4 = Add()([conv26, crop4])
    return Model(inputs=input1, outputs=eltwise4, name="cugan")

def upscale():
    parser = argparse.ArgumentParser(description="Upscale 200x200 image.")
    parser.add_argument("image", type=str, help="image path")
    parser.add_argument("-o", "--out", default="upscaled.jpg")
    parser.add_argument("-w", '--weights', default="models/weights_crop.h5")
    args = parser.parse_args()

    model = create_model()
    model.load_weights(args.weights)

    img = image.load_img(args.image)
    img_tensor = image.img_to_array(img)
    img_tensor = preprocess_input(img_tensor, data_format="channels_first", mode="torch")
    img_tensor = np.expand_dims(img_tensor, axis=0)

    pred = model.predict(img_tensor)
    pred = pred.reshape((332, 332, 3))

    image.save_img(args.out, pred)

    print(f"Created upscaled image: {args.out}")

if __name__ == '__main__':
    upscale()