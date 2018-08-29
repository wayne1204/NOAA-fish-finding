import json
import sys
import argparse
import os
import os.path as osp
import cv2
import xml.dom
import xml.dom.minidom
import numpy as np

from txtGenerator import generate_txt
from xmlGenerator import createRootNode, createObjectNode, writeXMLFile
from imgEnhance import enhance

def ArgumentParser():
    parser = argparse.ArgumentParser(description='Parse json annotations file and convert to VOC style')
    parser.add_argument('--dataset', choices=['mouss_seq0', 'mouss_seq1', 'mbari_seq0', 'habcam_seq0'])
    parser.add_argument('--mode', choices=['original', 'contrast', 'equal'], help='image enhancement')
    parser.add_argument('--output_path', default='data/VOCdevkit2007')
    parser.add_argument('--anno_path', help='training annotation files directory',
        default='/home/iis/Desktop/tf-faster-rcnn/data/NOAA/Training Data Release/annotations/')
    return parser.parse_args()

def getAnnotation(dataset, path):
    if args.dataset == 'mbari_seq0':
        train_anno = osp.join(path, '{}_new_training.mscoco.json'.format(dataset))
    else:
        train_anno = osp.join(path, '{}_training.mscoco.json'.format(dataset))
    return train_anno


args = ArgumentParser()

DATASET = args.dataset
IMG_PATH = osp.join(args.output_path, DATASET, 'PNGImages')
XML_PATH = osp.join(args.output_path, DATASET, 'Annotations')
if not os.path.exists(XML_PATH):
    os.mkdir(XML_PATH)

train_anno = getAnnotation(DATASET, args.anno_path)
with open(train_anno) as f:
    data = json.load(f)
    category = [c['name'] for c in data['categories']]
    images = data['images']
    annotation = data['annotations']

# id to image mapping
imageDict = {}
for image in images:
    if image['has_annots'] == True:
        key = image['id']
        imageDict[key] = image['file_name']

img = cv2.imread(osp.join(IMG_PATH, imageDict[key]))
height, width, channel = img.shape

total_annotation = {}
category_count = [0 for i in range(len(category))] 
counter = 0

for a in annotation:
    image_name = imageDict[a['image_id']].replace('.png', '')
    idx = a['category_id']-1
    single_ann = []
    single_ann.append(image_name)
    single_ann.append(category[idx])
    single_ann.extend(a['bbox'])

    if single_ann[4] != 0 and single_ann[5] != 0 and single_ann[2] < width and single_ann[3] < height:
        if image_name not in total_annotation:
            total_annotation[image_name] = []
        category_count[idx] += 1      
        total_annotation[image_name].append(single_ann)
    else:
        counter += 1


print('\n==============[ {} json info ]=============='.format(DATASET))
print("Total Annotations: {}".format(len(annotation)))
print("Total Image: {} / {}".format(len(total_annotation), len(images)))
print("wrong size Image: {}".format(counter))
print("Image shape: ({}, {})".format(width, height))
print("Total Category: {}".format(len(category)))
print("{:^20}| count".format("class"))
print('----------------------------')
for c, cnt in zip(category, category_count):
    if cnt != 0:
        print("{:^20}| {}".format(c, cnt))
fnames = list(total_annotation.keys())

for fname in fnames:
    is_save = True
    saveName = os.path.join(XML_PATH, fname + '.xml')
    doc, root_node = createRootNode(DATASET, fname, width, height, channel)

    for anno in total_annotation[fname]:
        object_node = createObjectNode(doc, anno, width, height)
        root_node.appendChild(object_node)    
        if args.dataset == 'habcam_seq0' and anno[1] in ['Cephalopoda', 'Aplousobranchia', 'Perciformes', 'Fish']:
            print("file {} with class {}".format(fname, anno[1]))
            is_save = False
            # os.system('cp {} {}'.format(os.path.join(IMG_PATH, fname + '.png'), os.path.join(clone_path, fname + '.png')))
    if is_save:
        writeXMLFile(doc, saveName)

generate_txt(DATASET)
mm = enhance(args.mode, IMG_PATH)
print('mean:', mm )
