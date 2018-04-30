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

fig = mlab.figure(bgcolor=(0, 0, 0), size=(256, 512))
for f in sorted(glob.glob('eval_kitti/build/results/%s/data/*.txt'%args.name)):
    idd = int(f.split('/')[-1].strip('.txt'))
    path = mapping[idd]
    print idd
    print path
    fr = int(path.strip().split(' ')[-1])
    basedir = args.fdir
    date = path.split(' ')[-3]
    drive = path.split(' ')[-2].split('_')[-2]

    dataset = pykitti.raw(basedir, date, drive,frames=range(fr, fr+1, 1))
    im = next(iter(itertools.islice(dataset.cam2, 0, None)))
    velo_pose = np.linalg.inv(dataset.calib.T_velo_imu)
    velo = next(iter(itertools.islice(dataset.velo, 0, None)))
    velo = velo_pose.dot(np.hstack( (velo[:,:3],np.ones((velo.shape[0],1)))).T).T
    cen = np.asarray([velo_pose.dot([0,0,0,1])])
    mlab.points3d(cen[:,0],cen[:,1],cen[:,2],color=(1,0,0),scale_factor=0.2)
    velo = velo[velo[:,0]>0]

    cam_pose = np.linalg.inv(dataset.calib.T_cam2_imu)
    filled_idx = np.zeros((velo.shape[0],),dtype=bool)
    predictions = np.loadtxt(f, delimiter=' ', usecols=(8, 9,10,11,12,13,14))
    label = np.loadtxt(f, delimiter=' ', usecols=(0),dtype=str)
    if predictions.ndim == 1:
        predictions = np.expand_dims(predictions,0)
        label = np.expand_dims(label,0)
    if predictions.size>0:  label = [l.lower() for l in label]
    for i,item in enumerate(predictions):
        if label[i] not in  class_map.keys(): continue
        translation =  item[3:6]
        h, w, l = item[:3]
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [0,0,0,0, -h, -h, -h, -h],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        ])
        
        yaw = item[6]
        rotMat = np.array([
                [np.cos(yaw),0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw),0.0, np.cos(yaw)],
               ])
        box = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
        im = draw_class.draw_projected_box3d(im,box[:].T, dataset.calib.P_rect_20,class_map[label[i]])
        box = cam_pose.dot( np.vstack((box, np.ones((1,8)) )) )
        draw_class.draw_box(box, class_map[label[i]])
        idx = in_hull(velo[:,:3],box[:3,:].T)
        draw_class.draw_cluster(velo[idx,:], class_map[label[i]])
        filled_idx |= idx
    draw_class.draw_cluster(velo[~filled_idx,:])

    gts = np.loadtxt('eval_kitti/build/results/gt/data/%06d.txt'%idd, delimiter=' ', usecols=(8, 9,10,11,12,13,14))
    gts_label = np.loadtxt('eval_kitti/build/results/gt/data/%06d.txt'%idd, delimiter=' ', usecols=(0),dtype=str)
    if gts.ndim <= 1:
        gts = np.expand_dims(gts,0)
        gts_label = np.expand_dims(gts_label,0)
    gts_label = [l.lower() for l in gts_label]
    for i,item in enumerate(gts):
        if gts_label[i] not in  class_map.keys(): continue
        translation =  item[3:6]
        h, w, l = item[:3]
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [0,0,0,0, -h, -h, -h, -h],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        ])

        yaw = item[6]
        rotMat = np.array([
                [np.cos(yaw),0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw),0.0, np.cos(yaw)],
               ])
        box = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
        im = draw_class.draw_projected_box3d(im,box[:].T, dataset.calib.P_rect_20,6)
        box = cam_pose.dot( np.vstack((box, np.ones((1,8)) )) )
        draw_class.draw_box(box,6)


    config=(180, 10, 120,[30,0,0 ])
    mlab.view(*config)
    mlab.savefig('./output/detection/%s/%05d.png'%(args.name,idd),magnification=4)
    #im = cv2.resize(im,(768,256))
    #plt.imsave('./output/detection/%s/%05d-cam.png'%(args.name,idd),im)
    cv2.imwrite('./output/detection/%s/%05d-cam.jpg'%(args.name,idd),im[:,:,::-1]*255)
    mlab.clf()
