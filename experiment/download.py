"""
Downloads our data sources, including the hyperspectral image, the canopy height model, and the tree labels.
"""

import os
import sys
import tqdm
import argparse

from wget import download

from .paths import *

parser = argparse.ArgumentParser()

args = parser.parse_args()

# make output directory if necessary
if not os.path.exists('data'):
    os.makedirs('data')

files = [ 'Labels_Trimmed_Selective.CPG',
          'Labels_Trimmed_Selective.dbf',
          'Labels_Trimmed_Selective.prj',
          'Labels_Trimmed_Selective.sbn',
          'Labels_Trimmed_Selective.sbx',
          'Labels_Trimmed_Selective.shp',
          'Labels_Trimmed_Selective.shp.xml',
          'Labels_Trimmed_Selective.shx',
          'NEON_D17_TEAK_DP1_20170627_181333_reflectance.tif',
          'NEON_D17_TEAK_DP1_20170627_181333_reflectance.tif.aux.xml',
          'NEON_D17_TEAK_DP1_20170627_181333_reflectance.tif.enp',
          'NEON_D17_TEAK_DP1_20170627_181333_reflectance.tif.ovr',
          'D17_CHM_all.tfw',
          'D17_CHM_all.tif',
          'D17_CHM_all.tif.aux.xml',
          'D17_CHM_all.tif.ovr',
        ]

for f in files:
  print('downloading %s'%f)
  download('https://zenodo.org/record/3468720/files/%s?download=1'%f,'data/%s'%f)
  print('')

