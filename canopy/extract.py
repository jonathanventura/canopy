import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import numpy as np
import tqdm

def extract_patches(image_uri,patch_radius,
                    chm_uri,height_threshold,
                    labels_uri):
    """Extract patches from an image

    At each labeled pixel, we extract a square patch around it.  We discard the patch if it contains "no data" pixels or if the height according to the CHM is below a threshold.
    Arguments:
      image_uri: URI for image
      patch_radius: radius of patch (e.g. radius of 7 = 15x15 patch)
      chm_uri: URI for canopy height model
      height_threshold: threshold below which pixels will be discarded
      label_uri: URI for the labels raster
    Returns:
      image patches, patch labels
    """

    # "no data value" for labels
    label_ndv = 255

    # open the hyperspectral image
    image = rasterio.open(image_uri)
    image_ndv = image.meta['nodata']
    image_width = image.meta['width']
    image_height = image.meta['height']

    # open the CHM
    chm = rasterio.open(chm_uri)
    chm_vrt = WarpedVRT(chm, crs=image.meta['crs'], transform=image.meta['transform'], width=image.meta['width'], height=image.meta['height'],
                       resampling=Resampling.bilinear)

    with rasterio.open(labels_uri,'r') as f:
        labels_raster = f.read(1)

    # create lists for the patches and labels
    image_patches = []
    patch_labels = []

    # get all labeled locations in the labels raster
    rows, cols = np.where(labels_raster!=label_ndv)

    # extract the patch for each location
    # tqdm makes the cool progress bar that you see
    with tqdm.tqdm(total=len(rows)) as pbar:

      for row, col in zip(rows, cols):

        # increment the progress bar
        pbar.update(1)

        # check height in canopy height model
        chm_val = chm_vrt.read(1,window=((row,row+1),(col,col+1)))
        if chm_val==chm.nodata or chm_val<=height_threshold: continue

        # check patch bounds against image bounds
        if row-patch_radius < 0 or col-patch_radius < 0: continue
        if row+patch_radius >= image_height or col+patch_radius >= image_width: continue

        # get patch from image
        image_patch = image.read(window=((row-patch_radius,row+patch_radius+1),(col-patch_radius,col+patch_radius+1)))

        # check for nodata in patch
        if np.any(image_patch<0): continue

        # append the patch and label to the lists
        image_patches.append(image_patch)
        patch_labels.append(labels_raster[row,col])

    # close the raster files
    image.close()
    chm.close()

    # stack the patches into a numpy array
    image_patches = np.stack(image_patches,axis=0)

    # re-order the dimensions so that we have (index,height,width,channels)
    image_patches = np.transpose(image_patches,axes=[0,2,3,1])

    # stack the labels into a numpy array
    patch_labels = np.stack(patch_labels,axis=0)

    return image_patches,patch_labels
