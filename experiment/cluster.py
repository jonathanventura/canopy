import numpy as np
import tqdm
import os
import h5py as h5

from canopy.vector_utils import *
from canopy.extract import *
from .paths import *

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import argparse

parser = argparse.ArgumentParser()

args = parser.parse_args()

np.random.seed(0)

# Load the metadata from the image.
with rasterio.open(image_uri) as src:
  image_meta = src.meta.copy()

# Load the shapefile and transform it to the hypersectral image's CRS.
polygons, labels = load_and_transform_shapefile(labels_shp_uri,'SP',image_meta['crs'])

# Cluster polygons for use in stratified sampling
centroids = np.stack([np.mean(np.array(poly['coordinates'][0]),axis=0) for poly in polygons])
cluster_ids = KMeans(10).fit_predict(centroids)
with open('data/clusters.txt','w') as f:
  for i in cluster_ids: f.write(str(i) + '\n')
rasterize_shapefile(polygons, cluster_ids, image_meta, './data/clusters.tiff')


