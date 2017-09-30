import random
import os
import sys
import csv
import time
import shutil
from glob import glob
from datetime import datetime
import joblib

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.initializers import he_uniform

from skimage.io import imread, imsave

from package.augmentation import random_rotation, random_shift, random_zoom, random_hsv, aug
from package.loss_and_metric import bce_dice_loss, dice_coef
from package.util import rle_encode

IMG_DIR = 'input/train_hq'
GT_DIR = 'input/train_masks'
TEST_IMG_DIR = 'input/test_hq'


def vgg_block(filters, layers, x_input):
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer=he_uniform())(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for _ in range(layers-1):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer=he_uniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def build_model():
    inputs = Input(shape=(640, 960, 3))

    conv1 = vgg_block(32, 2, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = vgg_block(64, 2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = vgg_block(128, 2, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = vgg_block(256, 2, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = vgg_block(512, 2, pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = vgg_block(1024, 2, pool5)

    up7 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=he_uniform())(conv6)
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    up7 = concatenate([up7, conv5], axis=3)
    conv7 = vgg_block(512, 2, up7)
    up8 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=he_uniform())(conv7)
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    up8 = concatenate([up8, conv4], axis=3)
    conv8 = vgg_block(256, 2, up8)
    up9 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=he_uniform())(conv8)
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)
    up9 = concatenate([up9, conv3], axis=3)
    conv9 = vgg_block(128, 2, up9)
    up10 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=he_uniform())(conv9)
    up10 = BatchNormalization()(up10)
    up10 = Activation('relu')(up10)
    up10 = concatenate([up10, conv2], axis=3)
    conv10 = vgg_block(64, 2, up10)
    up11 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=he_uniform())(conv10)
    up11 = BatchNormalization()(up11)
    up11 = Activation('relu')(up11)
    up11 = concatenate([up11, conv1], axis=3)
    conv11 = vgg_block(32, 2, up11)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=he_uniform())(conv11)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model


def generator(img_paths, gt_paths, batch_size, do_aug=True):
    while 1:
        rp = np.random.permutation(len(img_paths))
        img_paths = img_paths[rp]
        gt_paths = gt_paths[rp]

        for step in range(img_paths.shape[0] // batch_size):
            batch_img_paths = img_paths[step * batch_size:(step + 1) * batch_size]
            batch_gt_paths = gt_paths[step * batch_size:(step + 1) * batch_size]

            X_batch = []
            y_batch = []

            for img_path, gt_path in zip(batch_img_paths, batch_gt_paths):
                img = imread(img_path)
                img = cv2.resize(img, (960, 640))
                gt = imread(gt_path)
                gt = cv2.resize(gt, (960, 640))

                if do_aug:
                    img, gt = aug(img, gt)
                img = img.astype('float32') / 255
                gt = gt.astype('float32') / 255
                X_batch.append(img)
                y_batch.append(gt)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)[:, :, :, np.newaxis]

            yield X_batch, y_batch


def train(prms):
    model_name = 'model/unet12_0.5x'
    os.makedirs(model_name)
    joblib.dump(prms, os.path.join(model_name, 'prms.pkl'))

    img_paths = np.array(glob(os.path.join(IMG_DIR, '*.jpg')))
    train_img_paths, valid_img_paths = train_test_split(img_paths,
        test_size=0.2, random_state=41)
    train_gt_paths = []
    for img_path in train_img_paths:
        train_gt_paths.append(os.path.join(GT_DIR,
            os.path.splitext(os.path.basename(img_path))[0]+'_mask.gif'))
    train_gt_paths = np.array(train_gt_paths)
    valid_gt_paths = []
    for img_path in valid_img_paths:
        valid_gt_paths.append(os.path.join(GT_DIR,
            os.path.splitext(os.path.basename(img_path))[0]+'_mask.gif'))
    valid_gt_paths = np.array(valid_gt_paths)

    epochs = prms['epochs']
    batch_size = prms['batch_size']
    if prms['optimizer'] == 'Adam':
        optimizer =  Adam(lr=1e-4)
    elif prms['optimizer'] == 'RMSprop':
        optimizer = RMSprop(lr=1e-4)
    if prms['loss'] == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    elif prms['loss'] == 'bce_dice_loss':
        loss = bce_dice_loss

    model = build_model()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[dice_coef])
    callbacks = [
        EarlyStopping(min_delta=0, patience=20, verbose=1, mode='auto'),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, mode='auto',
            epsilon=1e-6, cooldown=0, min_lr=1e-5),
        ModelCheckpoint(os.path.join(model_name, 'model.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join(model_name, 'log.csv'))]

    model.fit_generator(generator(train_img_paths, train_gt_paths, batch_size=batch_size, do_aug=True),
        steps_per_epoch=len(train_img_paths) // batch_size,
        epochs=epochs,
        validation_data=generator(valid_img_paths, valid_gt_paths, batch_size=batch_size, do_aug=False),
        validation_steps=len(valid_img_paths) // batch_size,
        callbacks=callbacks,
        verbose=1)


def valid(model_name, pseudo=False):
    model = build_model()
    if not pseudo:
        model.load_weights(os.path.join('model', model_name, 'model.hdf5'))
    else:
        model.load_weights(os.path.join('model', model_name, 'model_pseudo.hdf5'))

    if not pseudo:
        valid_seg_dir = os.path.join('processed', '%s_valid'%model_name)
    else:
        valid_seg_dir = os.path.join('processed', '%s_valid_pseudo'%model_name)
    os.makedirs(valid_seg_dir)

    img_paths = np.array(glob(os.path.join(IMG_DIR, '*.jpg')))
    train_img_paths, valid_img_paths = train_test_split(img_paths,
        test_size=0.2, random_state=41)

    dices = []
    for i in tqdm(range(len(valid_img_paths))):
        img = imread(os.path.join(IMG_DIR,
            os.path.splitext(os.path.basename(valid_img_paths[i]))[0]+'.jpg'))
        img = cv2.resize(img, (960, 640))
        img = img.astype('float32') / 255
        mask = imread(os.path.join(GT_DIR,
            os.path.splitext(os.path.basename(valid_img_paths[i]))[0]+'_mask.gif'))
        mask = mask>127
        pb = model.predict(img[np.newaxis,:,:,:])
        pb = pb[0,:,:,0]
        pb = cv2.resize(pb, (1918, 1280))
        seg = pb>0.5
        dice = 2.0 * np.sum(seg&mask) / (np.sum(seg) + np.sum(mask))
        dices.append(dice)
        imsave(os.path.join(valid_seg_dir,
            os.path.splitext(os.path.basename(valid_img_paths[i]))[0]+'.png'), (pb*255).astype('uint8'))
    dices = np.array(dices)

    print(np.mean(dices))
    plt.hist(dices, bins=100)
    plt.xlabel('mean:%f'%np.mean(dices))
    if not pseudo:
        plt.savefig(os.path.join('model', model_name, 'valid_dice.png'))
    else:
        plt.savefig(os.path.join('model', model_name, 'valid_dice_pseudo.png'))


def test(model_name, pseudo=False):
    model = build_model()
    if not pseudo:
        model.load_weights(os.path.join('model', model_name, 'model.hdf5'))
    else:
        model.load_weights(os.path.join('model', model_name, 'model_pseudo.hdf5'))

    if not pseudo:
        test_seg_dir = os.path.join('processed', '%s_test'%model_name)
    else:
        test_seg_dir = os.path.join('processed', '%s_test_pseudo'%model_name)
    os.makedirs(test_seg_dir)

    y_test = pd.read_csv('input/sample_submission.csv')
    for i in tqdm(range(len(y_test))):
        img = imread(os.path.join('input/test_hq', y_test['img'][i]))
        img = cv2.resize(img, (960, 640))
        img = img.astype('float32') / 255
        pb = model.predict(img[np.newaxis,:,:,:])
        pb = pb[0,:,:,0]
        pb = cv2.resize(pb, (1918, 1280))
        y_test['rle_mask'][i] = rle_encode(pb>0.5)
        imsave(os.path.join(test_seg_dir,
            os.path.splitext(y_test['img'][i])[0]+'.png'), (pb*255).astype('uint8'))

    if not pseudo:
        y_test.to_csv(os.path.join('submission', '%s.csv.gz'%model_name), compression='gzip', index=False)
    else:
        y_test.to_csv(os.path.join('submission', '%s_pseudo.csv.gz'%model_name), compression='gzip', index=False)


def pseudo_labeling(model_name):
    # 0.4980
    model = build_model()
    model.load_weights(os.path.join('model', model_name, 'model.hdf5'))

    os.makedirs(os.path.join('processed', '%s_pseudo_label'%model_name))

    y_test = pd.read_csv('input/sample_submission.csv')
    img_paths = y_test['img'].as_matrix()
    rp = np.random.permutation(len(img_paths))
    img_paths = img_paths[rp]

    cnt = 0
    for i in range(len(img_paths)):
        img = imread(os.path.join('input/test_hq', img_paths[i]))
        img = cv2.resize(img, (960, 640))
        img = img.astype('float32') / 255
        pb = model.predict(img[np.newaxis,:,:,:])
        pb = pb[0,:,:,0]
        pb = cv2.resize(pb, (1918, 1280))
        mean_pb = np.mean(np.abs(pb.flatten() - 0.5))
        if mean_pb >= 0.4980:
            imsave(os.path.join('processed', '%s_pseudo_label'%model_name,
                os.path.splitext(os.path.basename(img_paths[i]))[0]+'.png'), (pb*255).astype('uint8'))
            cnt += 1
            if cnt%100 == 0:
                print('%d/%d'%(cnt, 2035))
        if cnt >= 2035:
            break


def pseudo_train(model_name, prms, pseudo_gt_dir):
    img_paths = glob(os.path.join(IMG_DIR, '*.jpg'))
    train_img_paths, valid_img_paths = train_test_split(img_paths,
        test_size=0.2, random_state=41)
    train_gt_paths = []
    for img_path in train_img_paths:
        train_gt_paths.append(os.path.join(GT_DIR,
            os.path.splitext(os.path.basename(img_path))[0]+'_mask.gif'))
    valid_gt_paths = []
    for img_path in valid_img_paths:
        valid_gt_paths.append(os.path.join(GT_DIR,
            os.path.splitext(os.path.basename(img_path))[0]+'_mask.gif'))
    valid_img_paths = np.array(valid_img_paths)
    valid_gt_paths = np.array(valid_gt_paths)

    pseudo_gt_paths = glob(os.path.join(pseudo_gt_dir, '*.png'))
    pseudo_img_paths = []
    for gt_path in pseudo_gt_paths:
        pseudo_img_paths.append(os.path.join(TEST_IMG_DIR,
            os.path.splitext(os.path.basename(gt_path))[0]+'.jpg'))
    train_img_paths.extend(pseudo_img_paths)
    train_img_paths = np.array(train_img_paths)
    train_gt_paths.extend(pseudo_gt_paths)
    train_gt_paths = np.array(train_gt_paths)

    epochs = prms['epochs']
    batch_size = prms['batch_size']
    if prms['optimizer'] == 'Adam':
        optimizer =  Adam(lr=1e-4)
    elif prms['optimizer'] == 'RMSprop':
        optimizer = RMSprop(lr=1e-4)
    if prms['loss'] == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    elif prms['loss'] == 'bce_dice_loss':
        loss = bce_dice_loss

    model = load_model(os.path.join('model', model_name, 'model.hdf5'),
        custom_objects={'dice_coef': dice_coef})
    callbacks = [
        EarlyStopping(min_delta=0, patience=10, verbose=1, mode='auto'),
        ModelCheckpoint(os.path.join('model', model_name, 'model_pseudo.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join('model', model_name, 'log_pseudo.csv'))]

    model.fit_generator(generator(train_img_paths, train_gt_paths, batch_size=batch_size, do_aug=True),
        steps_per_epoch=len(train_img_paths) // batch_size,
        epochs=epochs,
        validation_data=generator(valid_img_paths, valid_gt_paths, batch_size=batch_size, do_aug=False),
        validation_steps=len(valid_img_paths) // batch_size,
        callbacks=callbacks,
        verbose=1)


def main():
    prms = {
        'epochs':300,
        'batch_size':2,
        'optimizer':'Adam',
        'loss':'binary_crossentropy'
    }

    train(prms)
    #valid('unet12_0.5x', pseudo=True)
    pseudo_labeling('unet12_0.5x')
    pseudo_trian('unet12_0.5x', prms)
    test('unet12_0.5x', pseudo=True)

if __name__ == '__main__':
    main()
