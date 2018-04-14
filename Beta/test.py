import sys
import argparse
import numpy as np
from PIL import Image 
import requests
from io import BytesIO
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from keras.preprocessing import image
from keras.models import load_model

target_size = (256,256)
model = load_model('colornet1.h5')

def predict(img):
	#temp = img.resize(target_size)
	x = image.img_to_array(img)
	colorme = []
	colorme.append(x)
	colorme = np.array(colorme,dtype=float)
	colorme = rgb2lab(1.0/255*colorme)[:,:,:,0]
	colorme = colorme.reshape(colorme.shape+(1,))
	#model = load_model('/home/sanjit/CSI/Beta/model.h5')

	output = model.predict(colorme)
	output = output*128
	print (output)
	for i in range(len(output)):
		cur = np.zeros((256,256,3))
		cur[:,:,0] = colorme[i][:,:,0]
		cur[:,:,1:] = output[i]
		imsave("result.jpg",lab2rgb(cur))

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to image")
    args = a.parse_args()

    if args.image is None and args.image_url is None:
        a.print_help()
        sys.exit(1)

    if args.image is not None:
    	img = Image.open(args.image)
    	predict(img)