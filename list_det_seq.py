import os
#os.environ['QT_API']='pyqt'
#os.environ['ETS_TOOLKIT'] = 'qt4'
import time
import cv2
import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab
import time
from source.utils import load_tracklets_for_frames, point_inside, in_hull,mkdir_p
from source import parseTrackletXML as xmlParser
import argparse
from matplotlib import pyplot as plt
import Image
from cluster_pcs.filters import *
from math import atan2, degrees
import glob
#import pcl




class_map={'car':0,'pedestrian':4,'cyclist':2}
parser=argparse.ArgumentParser()
parser.add_argument('--fdir',type=str,help='dir of format base/data/drive',default='/data/KITTI-2/')
parser.add_argument('--name',type=str,help='dir of format base/data/drive',default='gt')
parser.add_argument('--outdir',type=str,help='output dir',default='/data/output')
args = parser.parse_args()

mkdir_p('./output/detection/%s'%(args.name))

with open('./det/mapping/train_mapping.txt','r') as f:
    lines = f.readlines()
with open('./det/mapping/train_rand.txt','r') as f:
    rands = f.readlines()[0].split(',')
mapping = {i:lines[int(r)-1] for i,r in enumerate(rands)}

for f in sorted(glob.glob('eval_kitti/build/results/%s/data/*.txt'%args.name)):
    idd = int(f.split('/')[-1].strip('.txt'))
    path = mapping[idd]
    fr = int(path.strip().split(' ')[-1])
    basedir = args.fdir
    date = path.split(' ')[-3]
    drive = path.split(' ')[-2].split('_')[-2]
    print '%s/%s/image_02/data/%010d.jpg' %(date, path.split(' ')[-2],fr)

