base_uri = './data'

# Canopy Height Model (CHM)
chm_uri = base_uri + '/D17_CHM_all.tif'

# Hyperspectral image
image_uri = base_uri + '/NEON_D17_TEAK_DP1_20170627_181333_reflectance.tif'
rgb_image_uri = base_uri + '/NEON_D17_TEAK_DP1QA_20170627_181333_RGB_Reflectance.tif'

# Labels shapefile
labels_shp_uri = base_uri + '/Labels_Trimmed_Selective.shp'

# Text files listing indices of train,val,test polygons (will be created)
train_ids_uri = 'train_ids.txt'
val_ids_uri = 'val_ids.txt'
test_ids_uri = 'test_ids.txt'

# Labels raster (will be created)
train_labels_uri = 'train_labels.tif'
val_labels_uri = 'val_labels.tif'
test_labels_uri = 'test_labels.tif'

# HDF5 data file for Keras (will be created)
train_data_uri = 'train_data.hdf5'
val_data_uri = 'val_data.hdf5'
test_data_uri = 'test_data.hdf5'

# estimated mean and std of training data (will be created)
mean_std_uri = 'train_mean_std.npz'

# weights for model (will be created)
weights_uri = 'weights.hdf5'

# prediction from model (will be created)
predict_uri = 'prediction.tiff'

