import numpy as np
import h5py as h5
from tqdm import tqdm, trange
import os
import sys

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from canopy.model import PatchClassifier
from .paths import *

import argparse

from tensorflow.keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--out',required=True,help='directory for output files')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--epochs',type=int,default=20,help='num epochs')
parser.add_argument('--norm',default='pca',help='normalization (meanstd or pca)')

args = parser.parse_args()

with h5.File(args.out + '/' + train_data_uri,'r') as f:
  x_all = f['data'][:].astype('float32')
  y_all = f['label'][:]

class_weights = compute_class_weight('balanced',range(8),y_all)
print('class weights: ',class_weights)
class_weight_dict = {}
for i in range(8):
  class_weight_dict[i] = class_weights[i]

def estimate_mean_std():
    x_samples = x_all[:,7,7]
    x_mean = np.mean(x_samples,axis=(0))
    x_std = np.std(x_samples,axis=(0))+1e-5
    return x_mean, x_std

def estimate_pca():
    x_samples = x_all[:,7,7]
    pca = PCA(32,whiten=True)
    pca.fit(x_samples)
    return pca

"""Normalize training data"""
if args.norm == 'meanstd':
    x_train_mean, x_train_std = estimate_mean_std()
    np.savez(args.out + '/' + mean_std_uri,x_train_mean,x_train_std)
elif args.norm == 'pca':
    pca = estimate_pca()
    dump(pca,args.out + '/pca.joblib')

x_shape = x_all.shape[1:]
x_dtype = x_all.dtype
y_shape = y_all.shape[1:]
y_dtype = y_all.dtype
if args.norm=='pca':
    x_shape = x_shape[:-1] + (pca.n_components_,)

print(x_shape, x_dtype)
print(y_shape, y_dtype)

classifier = PatchClassifier(num_classes=8)
model = classifier.get_patch_model(x_shape)

print(model.summary())

model.compile(optimizer=SGD(args.lr,momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def apply_pca(x):
  N,H,W,C = x.shape
  x = np.reshape(x,(-1,C))
  x = pca.transform(x)
  x = np.reshape(x,(-1,H,W,x.shape[-1]))
  return x

checkpoint = ModelCheckpoint(filepath=args.out + '/' + weights_uri, monitor='val_acc', verbose=True, save_best_only=True, save_weights_only=True)
reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

if args.norm == 'meanstd':
  x_all -= np.reshape(x_train_mean,(1,1,1,-1))
  x_all /= np.reshape(x_train_std,(1,1,1,-1))
elif args.norm == 'pca':
  x_all = apply_pca(x_all)

def augment_images(x,y):
  x_aug = []
  y_aug = []
  with tqdm(total=len(x)*8,desc='augmenting images') as pbar:
    for rot in range(4):
      for flip in range(2):
        for patch,label in zip(x,y):
          patch = np.rot90(patch,rot)
          if flip:
            patch = np.flip(patch,axis=0)
            patch = np.flip(patch,axis=1)
          x_aug.append(patch)
          y_aug.append(label)
          pbar.update(1)
  return np.stack(x_aug,axis=0), np.stack(y_aug,axis=0)

x_all, y_all = augment_images(x_all,y_all)

train_inds, val_inds = train_test_split(range(len(x_all)),test_size=0.1,random_state=0)
x_train = np.stack([x_all[i] for i in train_inds],axis=0)
y_train = np.stack([y_all[i] for i in train_inds],axis=0)
x_val = np.stack([x_all[i] for i in val_inds],axis=0)
y_val = np.stack([y_all[i] for i in val_inds],axis=0)

batch_size = 32

model.fit( x_train, y_train,
           epochs=args.epochs,
           batch_size=batch_size,
           validation_data=(x_val,y_val),
           verbose=1,
           callbacks=[checkpoint,reducelr],
           class_weight=class_weight_dict)

