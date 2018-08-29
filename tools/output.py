import os
import os.path as osp
import cv2
import argparse
import json

PATH = '/home/iis/Desktop/tf-faster-rcnn/data/NOAA/Training Data Release/annotations/'
# TEST_PATH = '/media/share33/Database/NOAA Fish Finding/Test Data/test_data'
TEST_PATH = '/home/iis/Desktop/tf-faster-rcnn/data/demo'


def ArgumentParser():
    parser = argparse.ArgumentParser(description='Parse json file')
    parser.add_argument(
        '--train_set', choices=['mouss_seq0', 'mouss_seq1', 'mbari_seq0', 'habcam_seq0', 'nwfsc_seq0', 'afsc_seq0'])
    parser.add_argument('--test_set', help='testing set')
    parser.add_argument(
        '--img_dict', default='data/fpaths_to_ids.json', help='image id mapping')
    parser.add_argument(
        '--output_dir', default='/home/iis/Desktop/NOAA_VOCdevkit/result')
    return parser.parse_args()


args = ArgumentParser()


def parseCategory(train_set):
    category_dict = {}
    train_anno = osp.join(PATH, '{}_training.mscoco.json'.format(train_set))
    with open(train_anno) as f:
        data = json.load(f)
        for item in data['categories']:
            category_dict[item['name']] = item['id']
    return category_dict


def parseImgDict(path):
    total_img = {}
    total_img['afsc'] = {}
    total_img['habcam'] = {}
    total_img['mbari1'] = {}
    total_img['mouss1'] = {}
    total_img['mouss2'] = {}
    total_img['mouss3'] = {}
    total_img['mouss4'] = {}
    total_img['mouss5'] = {}
    total_img['nwfsc'] = {}

    with open(path) as f:
        data = json.load(f)
        for key in data.keys():
            dataset, fname = key.split('/')
            total_img[dataset][fname] = data[key]
        for key in total_img.keys():
            print(key, len(total_img[key]))
        return total_img


def parseImg(data_root):
    total_img = []
    fnames = sorted(os.listdir(osp.join(TEST_PATH, args.test_set)))
    ss = cv2.imread(osp.join(TEST_PATH, args.test_set, fnames[1])).shape
    if "Thumbs.db" in fnames:
        fnames.remove('Thumbs.db')

    for cnt, fname in enumerate(fnames):
        img = {}

        img['id'] = cnt + 1
        img['file_name'] = fname
        img["width"] = ss[1]
        img["height"] = ss[0]
        img["has_annots"] = False
        total_img.append(img)
    data_root['image'] = total_img
    return fnames


def parseAnnotation(data_root, testset, img_dict, category):
    path = '/home/iis/Desktop/tf-faster-rcnn'
    with open(osp.join(path, testset + '_result.txt')) as f:
        annotations = [x.strip() for x in f.readlines()]
        for a in annotations:
            _object = {}
            aa = a.split()

            # image & category id
            _object["image_id"] = img_dict[aa[0]]
            _object["category_id"] = category[aa[1]]

            bbox = ([float(item) for item in aa[2:6]])
            bbox[3] -= bbox[1]
            bbox[2] -= bbox[0]
            _object["bbox"] = [round(i, 1) for i in bbox]
            _object["score"] = float(aa[-1])
            data_root.append(_object)



# if __name__ == '__main__':
data_root = []
category = parseCategory(args.train_set)
img_dict = parseImgDict(args.img_dict)
# imgs = parseImg(data_root)
if args.train_set in ['mouss_seq0', 'mouss_seq1', 'mbari_seq0', 'habcam_seq0']:
    parseAnnotation(data_root, args.test_set,
                    img_dict[args.test_set], category)

with open(osp.join(args.output_dir, args.test_set + '.mscoco.json'), 'w') as f:
    json.dump(data_root, f, indent=4)
