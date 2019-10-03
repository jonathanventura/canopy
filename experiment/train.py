import numpy as np

import h5py as h5
from tqdm import tqdm, trange

import os
import sys

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from canopy.model import PatchClassifier

from .paths import *

from sklearn.decomposition import PCA
from joblib import dump, load

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
parser.add_argument('--epochs',type=int,default=50,help='num epochs')
parser.add_argument('--norm',default='pca',help='normalization (meanstd or pca)')

args = parser.parse_args()

def estimate_mean_std(data_uri,num_samples=1000):
    with h5.File(data_uri,'r') as f:
        x_samples = f['data'][:].astype('float32')
        x_samples = x_samples[:,7,7]
        x_mean = np.mean(x_samples,axis=(0))
        x_std = np.std(x_samples,axis=(0))+1e-5
    return x_mean, x_std

def estimate_pca(data_uri,num_samples=1000):
    with h5.File(data_uri,'r') as f:
        x_samples = f['data'][:].astype('float32')
        print(x_samples.shape)
        x_samples = x_samples[:,7,7]
        print(x_samples.shape)
        pca = PCA(32,whiten=True)
        print('fitting PCA...')
        pca.fit(x_samples)
    return pca

"""Normalize training data"""
if args.norm == 'meanstd':
    x_train_mean, x_train_std = estimate_mean_std(args.out + '/' + train_data_uri)
    np.savez(args.out + '/' + mean_std_uri,x_train_mean,x_train_std)
elif args.norm == 'pca':
    pca = estimate_pca(args.out + '/' + train_data_uri)
    dump(pca,args.out + '/pca.joblib')

with h5.File(args.out + '/' + train_data_uri,'r') as f:
  x_shape = f['data'].shape[1:]
  x_dtype = f['data'].dtype
  y_shape = f['label'].shape[1:]
  y_dtype = f['label'].dtype
  if args.norm=='pca':
    x_shape = x_shape[:-1] + (pca.n_components_,)

  num_train = len(f['data'])

with h5.File(args.out + '/' + val_data_uri,'r') as f:
  num_val = len(f['data'])

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

def hdf5_generator(path,batch_size,shuffle=True,augment=True):
  with h5.File(path,'r') as f:
    x = f['data']
    y = f['label']
    while True:
      inds = np.random.randint(len(x),size=(batch_size))
      x_batch = []
      y_batch = []
      for i,ind in enumerate(inds):
        patch = x[ind]
        if augment:
          rot = np.random.randint(4)
          patch = np.rot90(patch,rot)
          if np.random.randint(2):
            patch = np.flip(patch,axis=0)
            patch = np.flip(patch,axis=1)
        x_batch.append(patch.astype('float32'))
        y_batch.append(y[ind])
      x_batch = np.stack(x_batch,axis=0)
      y_batch = np.stack(y_batch,axis=0)
      if args.norm == 'meanstd':
        x_batch -= np.reshape(x_train_mean,(1,1,1,-1))
        x_batch /= np.reshape(x_train_std,(1,1,1,-1))
      elif args.norm == 'pca':
        x_batch = apply_pca(x_batch)
      yield x_batch, y_batch

checkpoint = ModelCheckpoint(filepath=args.out + '/' + weights_uri, monitor='val_acc', verbose=True, save_best_only=True, save_weights_only=True)
reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

with h5.File(args.out + '/' + train_data_uri,'r') as f:
  x_train = f['data'][:].astype('float32')
  y_train = f['label'][:]

batch_size = 32

with h5.File(args.out + '/' + val_data_uri,'r') as f:
  x_val = f['data'][:].astype('float32')
  y_val = f['label'][:]

if args.norm == 'meanstd':
  x_train -= np.reshape(x_train_mean,(1,1,1,-1))
  x_train /= np.reshape(x_train_std,(1,1,1,-1))
  x_val -= np.reshape(x_train_mean,(1,1,1,-1))
  x_val /= np.reshape(x_train_std,(1,1,1,-1))
elif args.norm == 'pca':
  x_train = apply_pca(x_train)
  x_val = apply_pca(x_val)

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

x_train, y_train = augment_images(x_train,y_train)
x_val, y_val = augment_images(x_val,y_val)

model.fit( x_train, y_train,
                     epochs=args.epochs,
                     batch_size=batch_size,
                     validation_data=(x_val,y_val),
                     verbose=1,
                     callbacks=[checkpoint,reducelr])

