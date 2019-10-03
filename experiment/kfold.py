import numpy as np
import tqdm
import os
import h5py as h5

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from .paths import *

from canopy.vector_utils import *
from canopy.extract import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out',required=True,help='directory prefix for output files')
parser.add_argument('--rgb',action='store_true',help='use RGB data instead of hyperspectral')

args = parser.parse_args()

# make output directory if necessary
for i in range(10):
  if not os.path.exists(args.out+str(i)):
      os.makedirs(args.out+str(i))

if args.rgb:
    my_image_uri = rgb_image_uri
else:
    my_image_uri = image_uri

# Load the metadata from the image.
with rasterio.open(my_image_uri) as src:
  image_meta = src.meta.copy()

# Load the shapefile and transform it to the hypersectral image's CRS.
polygons, labels = load_and_transform_shapefile(labels_shp_uri,'SP',image_meta['crs'])

cluster_ids = np.loadtxt('./data/clusters.txt',dtype='int')

for fold in range(10):
  os.makedirs(args.out + str(fold),exist_ok=True)
  with open(args.out + str(fold) + '/' + train_ids_uri,'w') as f:
    pass
  with open(args.out + str(fold) + '/' + test_ids_uri,'w') as f:
    pass

kfold = KFold(n_splits=10, shuffle=True, random_state=0)
for cluster in range(10):
  subset = np.where(cluster_ids==cluster)[0]

  fold = 0
  for train_idx, test_idx in kfold.split(X=subset):
    train_inds = subset[train_idx]
    test_inds = subset[test_idx]
    # Save ids of train,val,test polygons
    with open(args.out + str(fold) + '/' + train_ids_uri,'a') as f:
      f.writelines(["%d\n"%ind for ind in train_inds])
    with open(args.out + str(fold) + '/' + test_ids_uri,'a') as f:
      f.writelines(["%d\n"%ind for ind in test_inds])
    fold = fold + 1

for fold in range(10):
  train_inds = np.loadtxt(args.out + str(fold) + '/' + train_ids_uri,dtype='int')
  test_inds = np.loadtxt(args.out + str(fold) + '/' + test_ids_uri,dtype='int')

  # Separate out polygons
  train_polygons = [polygons[ind] for ind in train_inds]
  train_labels = [labels[ind] for ind in train_inds]
  test_polygons = [polygons[ind] for ind in test_inds]
  test_labels = [labels[ind] for ind in test_inds]

  # Rasterize the shapefile to a TIFF.  Using LZW compression, the resulting file is pretty small.
  train_labels_raster = rasterize_shapefile(train_polygons, train_labels, image_meta, args.out + str(fold) + '/' + train_labels_uri)
  test_labels_raster = rasterize_shapefile(test_polygons, test_labels, image_meta, args.out + str(fold) + '/' + test_labels_uri)

  # Extract patches and labels
  patch_radius = 7
  height_threshold = 5
  train_image_patches, train_patch_labels = extract_patches(my_image_uri,patch_radius,chm_uri,height_threshold,args.out + str(fold) + '/' + train_labels_uri)
  test_image_patches, test_patch_labels = extract_patches(my_image_uri,patch_radius,chm_uri,height_threshold,args.out + str(fold) + '/' + test_labels_uri)

  # Store the patches and labels into an HDF5 file for easy use in Keras.
  with h5.File(args.out + str(fold) + '/' + train_data_uri,'w') as f:
    f.create_dataset('data',data=train_image_patches)
    f.create_dataset('label',data=train_patch_labels)
  with h5.File(args.out  + str(fold)+ '/' + test_data_uri,'w') as f:
    f.create_dataset('data',data=test_image_patches)
    f.create_dataset('label',data=test_patch_labels)

