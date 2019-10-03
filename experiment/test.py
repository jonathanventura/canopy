import numpy as np
import cv2
from math import floor, ceil
import tqdm
from joblib import dump, load

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from canopy.model import PatchClassifier
from .paths import *

import argparse

from tensorflow.keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--out',required=True,help='directory for output files')
parser.add_argument('--rgb',action='store_true',help='use RGB data instead of hyperspectral')
parser.add_argument('--norm',default='pca',help='normalization (meanstd or pca)')
args = parser.parse_args()

if args.norm == 'meanstd':
  with np.load(args.out + '/' + mean_std_uri) as f:
      x_train_mean = f['arr_0']
      x_train_std = f['arr_1']
elif args.norm == 'pca':
  pca = load(args.out + '/pca.joblib')

# "no data value" for labels
label_ndv = 255

# radius of square patch (side of patch = 2*radius+1)
patch_radius = 7

# height threshold for CHM -- pixels at or below this height will be discarded
height_threshold = 5

# tile size for processing
tile_size = 128

# tile size with padding
padded_tile_size = tile_size + 2*patch_radius

# open the hyperspectral or RGB image
if args.rgb:
  image = rasterio.open(rgb_image_uri)
else:
    image = rasterio.open(image_uri)
image_meta = image.meta.copy()
image_ndv = image.meta['nodata']
image_width = image.meta['width']
image_height = image.meta['height']
image_channels = image.meta['count']

# load model
if args.norm == 'pca':
  input_shape = (padded_tile_size,padded_tile_size,pca.n_components_)
else:
  input_shape = (padded_tile_size,padded_tile_size,image_channels)
tree_classifier = PatchClassifier(num_classes=8)
training_model = tree_classifier.get_patch_model(input_shape)
training_model.load_weights(args.out + '/' + weights_uri)
model = tree_classifier.get_convolutional_model(input_shape)

# calculate number of tiles
num_tiles_y = ceil(image_height / float(tile_size))
num_tiles_x = ceil(image_width / float(tile_size))

print('Metadata for image')
for key in image_meta.keys():
  print('%s:'%key)
  print(image_meta[key])
  print()

# create predicted label raster
predict_meta = image_meta.copy()
predict_meta['dtype'] = 'uint8'
predict_meta['nodata'] = label_ndv
predict_meta['count'] = 1
predict = rasterio.open(args.out + '/' + predict_uri, 'w', compress='lzw', **predict_meta)

# open the CHM
chm = rasterio.open(chm_uri)
chm_vrt = WarpedVRT(chm, crs=image.meta['crs'], transform=image.meta['transform'], width=image.meta['width'], height=image.meta['height'],
                   resampling=Resampling.bilinear)

# dilation kernel
kernel = np.ones((patch_radius*2+1,patch_radius*2+1),dtype=np.uint8)

def apply_pca(x):
  N,H,W,C = x.shape
  x = np.reshape(x,(-1,C))
  x = pca.transform(x)
  x = np.reshape(x,(-1,H,W,x.shape[-1]))
  return x

# go through all tiles of input image
# run convolutional model on tile
# write labels to output label raster
with tqdm.tqdm(total=num_tiles_y*num_tiles_x) as pbar:
    for y in range(patch_radius,image_height-patch_radius,tile_size):
        for x in range(patch_radius,image_width-patch_radius,tile_size):
            pbar.update(1)

            window = Window(x-patch_radius,y-patch_radius,padded_tile_size,padded_tile_size)

            # get tile from chm
            chm_tile = chm_vrt.read(1,window=window)
            if chm_tile.shape[0] != padded_tile_size or chm_tile.shape[1] != padded_tile_size:
              pad = ((0,padded_tile_size-chm_tile.shape[0]),(0,padded_tile_size-chm_tile.shape[1]))
              chm_tile = np.pad(chm_tile,pad,mode='constant',constant_values=0)
          
            chm_tile = np.expand_dims(chm_tile,axis=0)
            chm_bad = chm_tile <= height_threshold

            # get tile from image
            image_tile = image.read(window=window)
            image_pad_y = padded_tile_size-image_tile.shape[1]
            image_pad_x = padded_tile_size-image_tile.shape[2]
            output_window = Window(x,y,tile_size-image_pad_x,tile_size-image_pad_y)
            if image_tile.shape[1] != padded_tile_size or image_tile.shape[2] != padded_tile_size:
              pad = ((0,0),(0,image_pad_y),(0,image_pad_x))
              image_tile = np.pad(image_tile,pad,mode='constant',constant_values=-1)

            # re-order image tile to have height,width,channels
            image_tile = np.transpose(image_tile,axes=[1,2,0])

            # add batch axis
            image_tile = np.expand_dims(image_tile,axis=0)
            image_bad = np.any(image_tile < 0,axis=-1)

            image_tile = image_tile.astype('float32')
            if args.norm == 'meanstd':
              # remove mean and std. dev.
              image_tile -= np.reshape(x_train_mean,(1,1,1,-1))
              image_tile /= np.reshape(x_train_std,(1,1,1,-1))
            elif args.norm == 'pca':
              image_tile = apply_pca(image_tile)
            
            # run tile through network
            predict_tile = np.argmax(model.predict(image_tile),axis=-1).astype('uint8')

            # dilate mask
            image_bad = cv2.dilate(image_bad.astype('uint8'),kernel).astype('bool')

            # set bad pixels to NDV
            predict_tile[chm_bad[:,patch_radius:-patch_radius,patch_radius:-patch_radius]] = label_ndv
            predict_tile[image_bad[:,patch_radius:-patch_radius,patch_radius:-patch_radius]] = label_ndv

            # undo padding
            if image_pad_y > 0:
              predict_tile = predict_tile[:,:-image_pad_y,:]
            if image_pad_x > 0:
              predict_tile = predict_tile[:,:,:-image_pad_x]

            # write to file
            predict.write(predict_tile,window=output_window)

image.close()
chm.close()
predict.close()
