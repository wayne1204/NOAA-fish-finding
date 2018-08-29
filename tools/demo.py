#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

import xml.etree.ElementTree as ET
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


CLASSES = {'mbari': ('__background__', 'NonLiving', 'Scorpaeniformes', 'Pleuronectiformes', 
                    'Echinodermata', 'Gadiformes', 'Perciformes', 'Animalia', 'Aplousobranchia',
                    'Carcharhiniformes', 'Cephalopoda', 'Chordata', 'Decapoda', 'Fish', 
                    'Gastropoda', 'Mollusca', 'NotPleuronectiformes', 'Osmeriformes',
                    'Osteroida', 'Physical', 'Rajiformes', 'ignore'),
           'mouss0': ('__background__', 'Carcharhiniformes', 'Animalia', 'Aplousobranchia', 'Cephalopoda',
                      'Chordata', 'Decapoda', 'Echinodermata', 'Fish', 'Gadiformes', 'Gastropoda', 'Mollusca',
                      'NonLiving', 'NotPleuronectiformes', 'Osmeriformes', 'Osteroida', 'Perciformes',
                      'Physical', 'Pleuronectiformes', 'Rajiformes', 'Scorpaeniformes', 'ignore'), 
           'mouss1': ('__background__', 'Perciformes', 'Animalia', 'Aplousobranchia', 'Carcharhiniformes',
                     'Cephalopoda', 'Chordata', 'Decapoda', 'Echinodermata', 'Fish', 'Gadiformes', 
                     'Gastropoda', 'Mollusca', 'NonLiving', 'NotPleuronectiformes', 
                     'Osmeriformes', 'Osteroida', 'Physical', 'Pleuronectiformes', 
                     'Rajiformes', 'Scorpaeniformes', 'ignore'),
           'habcam': ('__background__', 'Gastropoda', 'Osteroida', 'Cephalopoda', 'Decapoda', 'Aplousobranchia',
                      'NotPleuronectiformes', 'Perciformes', 'Fish', 'Rajiformes', 'NonLiving', 'Pleuronectiformes',
                      'Animalia', 'Carcharhiniformes', 'Chordata', 'Echinodermata', 'Gadiformes', 'Mollusca',
                      'Osmeriformes', 'Physical', 'Scorpaeniformes', 'ignore'),
           }

DATASETS = {'mouss0':'mouss_seq0_trainval', 'mouss1': 'mouss_seq1_trainval', 'mbari': 'mbari_seq0_trainval', 'habcam': 'habcam_seq0_train'}
ITERS = {'mouss0': 30000, 'mouss1': 30000, 'mbari': 70000, 'habcam': 200000}
NETS = '{}_faster_rcnn_iter_{}.ckpt'

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=['vgg16', 'res101'], default='res101')
    parser.add_argument('--train_set',  help='Trained dataset [mouss0/1 mbari habcam]',
                        choices=DATASETS.keys(), default='mbari')
    parser.add_argument('--test_set', help='Tested dataset [mouss1/2/3/4/5 mbari1 habcam]',
                        default='test')
    parser.add_argument('--mode', help='mode to use. both: plot ground truth & predict, predict: plot predict bbox',
                         choices=['predict', 'both'], default='both')
    args = parser.parse_args()

    return args

def plot_groundtruth(ax, ann_file):
    tree = ET.parse(ann_file) 
    for child in tree.getroot(): 
        if child.tag == 'object':
            class_name = child[0].text
            bbox = [float(child[4][i].text) for i in range(4)]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='yellow', linewidth=1.5)
            )
            # ax.text(bbox[0], bbox[3],
            #         '{:s}'.format(class_name),
            #         bbox=dict(facecolor='blue', alpha=0.3),
            #         fontsize=6, color='white')
    plt.axis('off')


def vis_detections(ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] > thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.5)
            )
        ax.text(bbox[0], bbox[1],
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.3),
                fontsize=6, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, mode, net, dataset, test_set, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file  = os.path.join(cfg.DATA_DIR, 'demo', test_set ,image_name)
    ann_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007', DATASETS[dataset].replace('_trainval', ''),
                           'Annotations', image_name.replace('jpg', 'xml'))

    # Visualize detections for each class
    CONF_THRESH = 0.001
    NMS_THRESH = 0.3

    im = cv2.imread(im_file)
    im = im[:, :, (2, 1, 0)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, aspect='equal')

    if args.mode == 'both':
        ax.set_title('ground truth {}'.format(image_name), fontsize=10)
        plot_groundtruth(ax, ann_file)

    if args.mode != 'truth':
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, im)
        timer.toc()
        # ax.set_title('predict {}'.format(image_name), fontsize=10)
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        # filtered bbox by confidence & NMS threshold
        for cls_ind, category in enumerate(CLASSES[dataset][1:]):
            # because we skipped background
            cls_ind += 1 
            inds = np.where(scores[:, cls_ind] > CONF_THRESH)[0]

            cls_boxes = boxes[inds, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[inds, cls_ind]
            # dets : (300 x 5)
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            for k in range(dets.shape[0]):
                # output img / class / bbox / score
                with open('result/{}_result.txt'.format(test_set), 'a') as f:
                    f.write('{:s} {:s} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(
                        image_name, category,
                        dets[k, 0] + 1, dets[k, 1] + 1,
                        dets[k, 2] + 1, dets[k, 3] + 1,
                        dets[k, -1]))
            vis_detections(ax, category, dets)
        plt.tight_layout()
        

if __name__ == '__main__':  
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.train_set
    test_set = args.test_set
    iteration = ITERS[dataset]
    tfmodel = os.path.join('output', demonet, DATASETS[dataset], 'default',
                           NETS.format(demonet, iteration))

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # init
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 22,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    total_time = 0.0
    im_names = os.listdir(os.path.join('data', 'demo', args.test_set))
    if args.mode == 'truth':
        save_path = os.path.join('data', 'bbox', args.test_set + '_gt')
    else:
        save_path = os.path.join('data', 'bbox', args.test_set + '_' + demonet)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, im_name in enumerate(im_names):
        if im_name[-3:] in ['png', 'jpg']:
            print('==========================================')
            print('Dataset: {}'.format(args.test_set))
            print('Images: {}'.format(im_name))
            demo(sess, args.mode, net, dataset, test_set, im_name)
            plt.savefig(os.path.join(save_path, im_name))
            plt.cla()
            
          
