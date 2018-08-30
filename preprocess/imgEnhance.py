import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

def ArgumentParser():
    parser = argparse.ArgumentParser(description='Parse json file')
    parser.add_argument('--dataset', choices=['mouss_seq0', 'mouss_seq1', 'mbari_seq0', 'habcam_seq0'])
    parser.add_argument('--mode', choices=['original', 'contrast', 'equal'], default='original')
    parser.add_argument('--output_dir', default='data/VOCdevkit2007')
    return parser.parse_args()


def enhance(mode, img_path):
    fnames = os.listdir(img_path.replace('PNGImages', 'Annotations'))
    OUT_PATH = img_path.replace('PNGImages', 'JPEGImages')
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    total_mean = []
    for idx, fname in enumerate(fnames):
        print('enhance image{:5d}/{}'.format(idx+1, len(fnames)), end="\r")
        img_name = os.path.join(img_path, fname.replace('.xml', '.png'))
        if mode == 'original':
            img = cv2.imread(img_name)
            mean = np.mean(img, axis=(0, 1))
            total_mean.append(mean)
            # print(mean)
            cv2.imwrite(os.path.join(OUT_PATH, fname.replace('.xml', '.jpg')), img)

        elif mode == 'contrast':
            im = Image.open(img_name)
            en = ImageEnhance.Contrast(im)
            image = en.enhance(1.5)
            image.save(os.path.join(OUT_PATH, fname.replace('.xml', '.jpg')))
            mean = np.mean(image, axis=(0, 1))
            total_mean.append(mean)
            print(mean)

        elif mode == 'equal':
            img = cv2.imread(img_name)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            mean = np.mean(img_output, axis=(0, 1))
            total_mean.append(mean)
            print(mean)
            # for i, col in enumerate(color):
            #     histr = cv2.calcHist([img_output], [i], None, [256], [0, 256])
            #     plt.plot(histr,color = col)
            #     plt.xlim([0,256])
            # plt.show()

            # cv2.imwrite(os.path.join(fname.replace('.xml', '.jpg')), img)
            cv2.imwrite(os.path.join(OUT_PATH, fname.replace('.xml', '.jpg')), img_output)
    return np.mean(total_mean, axis=0)

if __name__ == '__main__':
    args = ArgumentParser()
    IMG_PATH = os.path.join(args.output_dir, args.dataset, 'PNGImages')
    XML_PATH = os.path.join(args.output_dir, args.dataset, 'Annotations')
    mm = enhance(args.mode, IMG_PATH)
