import os
#os.environ['QT_API']='pyqt'
#os.environ['ETS_TOOLKIT'] = 'qt4'
import time
import itertools
import pdb
import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab
import time
from source.utils import load_tracklets_for_frames, point_inside, in_hull
from source import parseTrackletXML as xmlParser
import argparse
from matplotlib import cm
from cluster_pcs.filters import *
from math import atan2, degrees
import glob
#import pcl

class_map={'Car':0,'Pedestrian':4,'Cyclist':2}
parser=argparse.ArgumentParser()
parser.add_argument('--fdir',type=str,help='dir of format base/data/drive',default='/data/KITTI-2/')
parser.add_argument('--data',type=str,help='dir of format base/data/drive',default='eval_kitti/build/results/gt/data/')
parser.add_argument('--outdir',type=str,help='output dir',default='/data/output')
args = parser.parse_args()

with open('./det/mapping/train_mapping.txt','r') as f:
    lines = f.readlines()
with open('./det/mapping/train_rand.txt','r') as f:
    rands = f.readlines()[0].split(',')
mapping = {i:lines[int(r)-1] for i,r in enumerate(rands)}

fig = mlab.figure(bgcolor=(0, 0, 0), size=(256, 512))
for npic,f in enumerate(sorted(glob.glob('%s/*.txt'%args.data))):
    idd = int(f.split('/')[-1].strip('.txt'))
    path = mapping[idd]
    print idd
    print path
    fr = int(path.strip().split(' ')[-1])
    basedir = args.fdir
    date = path.split(' ')[-3]
    drive = path.split(' ')[-2].split('_')[-2]

    dataset = pykitti.raw(basedir, date, drive,frames=range(fr, fr+1, 1))
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
        box = cam_pose.dot( np.vstack((box, np.ones((1,8)) )) )
        draw_class.draw_box(box, class_map[label[i]])
        idx = in_hull(velo[:,:3],box[:3,:].T)
        draw_class.draw_cluster(velo[idx,:], class_map[label[i]])
        filled_idx |= idx
    draw_class.draw_cluster(velo[~filled_idx,:])
    config=(180, 10, 100,[27,0,0 ])
    mlab.view(*config)
    mlab.savefig('./output/detection/%05d.png'%(idd),magnification=4)
    mlab.clf()
