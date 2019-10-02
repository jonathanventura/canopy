"""
Loads and co-register our data sources, including the hyperspectral image, the canopy height model, and the tree labels.  Then we build a dataset of patches and their corresponding labels and store it in a HDF5 file for easy use in Keras.

We are splitting the tree polygons into train/val/test splits (rather than splitting the patches) for a fairer evaluation methodology.
"""

import numpy as np
import tqdm
from .paths import *
import os

from canopy.vector_utils import *
from canopy.extract import *
import h5py as h5

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import argparse

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out',required=True,help='directory for output files')
parser.add_argument('--rgb',action='store_true',help='use RGB data instead of hyperspectral')
parser.add_argument('--seed',type=int,default=1234,help='random seed for train val test split')

args = parser.parse_args()

# make output directory if necessary
if not os.path.exists(args.out):
    os.makedirs(args.out)

if args.rgb:
    my_image_uri = rgb_image_uri
else:
    my_image_uri = image_uri

# Load the metadata from the image.
with rasterio.open(my_image_uri) as src:
  image_meta = src.meta.copy()

# Load the shapefile and transform it to the hypersectral image's CRS.
polygons, labels = load_and_transform_shapefile(labels_shp_uri,'SP',image_meta['crs'])

# Cluster polygons for use in stratified sampling
centroids = np.stack([np.mean(np.array(poly['coordinates'][0]),axis=0) for poly in polygons])
cluster_ids = KMeans(10).fit_predict(centroids)
rasterize_shapefile(polygons, cluster_ids, image_meta, args.out + '/clusters.tiff')
stratify = cluster_ids

# alternative: stratify by species label
# stratify = labels

# Split up polygons into train, val, test here
train_inds, test_inds = train_test_split(range(len(polygons)),test_size=0.1,random_state=args.seed,stratify=stratify)
train_inds, val_inds = train_test_split(train_inds,test_size=0.1,random_state=args.seed,stratify=[stratify[ind] for ind in train_inds])
print(len(train_inds),len(val_inds),len(test_inds))

# Save ids of train,val,test polygons
with open(args.out + '/' + train_ids_uri,'w') as f:
    f.writelines(["%d\n"%ind for ind in train_inds])
with open(args.out + '/' + val_ids_uri,'w') as f:
    f.writelines(["%d\n"%ind for ind in val_inds])
with open(args.out + '/' + test_ids_uri,'w') as f:
    f.writelines(["%d\n"%ind for ind in test_inds])

# Separate out polygons
train_polygons = [polygons[ind] for ind in train_inds]
train_labels = [labels[ind] for ind in train_inds]
val_polygons = [polygons[ind] for ind in val_inds]
val_labels = [labels[ind] for ind in val_inds]
test_polygons = [polygons[ind] for ind in test_inds]
test_labels = [labels[ind] for ind in test_inds]

# Rasterize the shapefile to a TIFF.  Using LZW compression, the resulting file is pretty small.
train_labels_raster = rasterize_shapefile(train_polygons, train_labels, image_meta, args.out + '/' + train_labels_uri)
val_labels_raster = rasterize_shapefile(val_polygons, val_labels, image_meta, args.out + '/' + val_labels_uri)
test_labels_raster = rasterize_shapefile(test_polygons, test_labels, image_meta, args.out + '/' + test_labels_uri)

# Extract patches and labels
patch_radius = 7
height_threshold = 5
train_image_patches, train_patch_labels = extract_patches(my_image_uri,patch_radius,chm_uri,height_threshold,args.out + '/' + train_labels_uri)
val_image_patches, val_patch_labels = extract_patches(my_image_uri,patch_radius,chm_uri,height_threshold,args.out + '/' + val_labels_uri)
test_image_patches, test_patch_labels = extract_patches(my_image_uri,patch_radius,chm_uri,height_threshold,args.out + '/' + test_labels_uri)

# Store the patches and labels into an HDF5 file for easy use in Keras.
with h5.File(args.out + '/' + train_data_uri,'w') as f:
  f.create_dataset('data',data=train_image_patches)
  f.create_dataset('label',data=train_patch_labels)
with h5.File(args.out + '/' + val_data_uri,'w') as f:
  f.create_dataset('data',data=val_image_patches)
  f.create_dataset('label',data=val_patch_labels)
with h5.File(args.out + '/' + test_data_uri,'w') as f:
  f.create_dataset('data',data=test_image_patches)
  f.create_dataset('label',data=test_patch_labels)
