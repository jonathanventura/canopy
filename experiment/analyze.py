import numpy as np

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.mask import mask

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import mapping

import tqdm

from math import floor, ceil

from .paths import *

from canopy.vector_utils import *
from canopy.extract import *

import sklearn.metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out',required=True,help='directory for output files')
parser.add_argument('--rgb',action='store_true',help='use RGB data instead of hyperspectral')
parser.add_argument('--rf',action='store_true',help='test random forest prediction')
args = parser.parse_args()

if args.rgb:
    my_image_uri = rgb_image_uri
else:
    my_image_uri = image_uri

train_inds = np.loadtxt(args.out + '/' + train_ids_uri,dtype='int32')
test_inds = np.loadtxt(args.out + '/' + test_ids_uri,dtype='int32')

# Load the metadata from the image.
with rasterio.open(my_image_uri) as src:
  image_meta = src.meta.copy()

# Load the shapefile and transform it to the hypersectral image's CRS.
polygons, labels = load_and_transform_shapefile(labels_shp_uri,'SP',image_meta['crs'])

train_labels = [labels[ind] for ind in train_inds]
test_labels = [labels[ind] for ind in test_inds]

# open predicted label raster
if args.rf:
    print('***** rf is on ******')
    predict = rasterio.open(args.out + '/rf_' + predict_uri)
else:
    predict = rasterio.open(args.out + '/' + predict_uri)
predict_raster = predict.read(1)
ndv = predict.meta['nodata']

def get_predictions(inds):
    preds = []
    for ind in inds:
        poly = [mapping(Polygon(polygons[ind]['coordinates'][0]))]
        out_image, out_transform = mask(predict, poly, crop=False)
        out_image = out_image[0]
        
        label = labels[ind]

        rows, cols = np.where(out_image != ndv)
        predict_labels = []
        for row, col in zip(rows,cols):
            predict_labels.append(predict_raster[row,col])
        predict_labels = np.array(predict_labels)
        
        hist = [np.count_nonzero(predict_labels==i) for i in range(8)]
        majority_label = np.argmax(hist)
        preds.append(majority_label)
    return preds

def calculate_confusion_matrix(labels,preds):
    mat = np.zeros((8,8),dtype='int32')
    for label,pred in zip(labels,preds):
        mat[label,pred] += 1
    return mat

def calculate_fscore(labels,preds):
  return sklearn.metrics.f1_score(labels,preds,average='micro')

test_preds = get_predictions(test_inds)
 
report = classification_report(test_labels, test_preds)
mat = confusion_matrix(test_labels,test_preds)
print('classification report:')
print(report)
print('confusion matrix:')
print(mat)
if args.rf:
  with open(args.out + '/rf_report.txt','w') as f:
    f.write(report)
  np.savetxt(args.out + '/rf_labels.txt',test_labels,delimiter=',')
  np.savetxt(args.out + '/rf_preds.txt',test_preds,delimiter=',')
  np.savetxt(args.out + '/rf_confusion.txt',mat,delimiter=',')
else:
  with open(args.out + '/report.txt','w') as f:
    f.write(report)
  np.savetxt(args.out + '/labels.txt',test_labels,delimiter=',')
  np.savetxt(args.out + '/preds.txt',test_preds,delimiter=',')
  np.savetxt(args.out + '/confusion.txt',mat,delimiter=',')
