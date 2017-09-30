import random
import os
import sys
import csv
import time
import shutil
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd

from skimage.io import imread, imsave

from package.util import rle_encode


y_test = pd.read_csv('input/sample_submission.csv')
for i in tqdm(range(len(y_test))):
    pb1 = imread(os.path.join('processed/unet12_0.5x_test_pseudo',
    os.path.splitext(os.path.basename(y_test['img'][i]))[0]+'.png'))
    pb1 = pb1.astype('float32') / 255
    pb2 = imread(os.path.join('processed/unet14_0.5x_test_pseudo',
        os.path.splitext(os.path.basename(y_test['img'][i]))[0]+'.png'))
    pb2 = pb2.astype('float32') / 255
    pb3 = imread(os.path.join('processed/unet12_1x_test_pseudo',
        os.path.splitext(os.path.basename(y_test['img'][i]))[0]+'.png'))
    pb3 = pb3.astype('float32') / 255
    pb = (pb1 + pb2 + pb3) / 3
    y_test['rle_mask'][i] = rle_encode(pb>0.5)

y_test.to_csv(os.path.join('submission', 'avg_unet12_0.5x_pseudo_unet14_0.5x_pseudo_unet12_1x_pseudo.csv.gz'), compression='gzip', index=False)
