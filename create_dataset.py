import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil

CUR_DIR = os.path.dirname(__file__)
NEW_DATASET = os.path.join(CUR_DIR, "new_dataset")

VOC_ROOT = os.path.join(CUR_DIR, "VOC2012")
TRAIN_DIR = os.path.join(NEW_DATASET, "train")
TRAIN_A_DIR = os.path.join(NEW_DATASET, "trainannot")
VALID_DIR = os.path.join(NEW_DATASET, "valid")
VALID_A_DIR = os.path.join(NEW_DATASET, "validannot")
TEST_DIR = os.path.join(NEW_DATASET, "test")
TEST_A_DIR = os.path.join(NEW_DATASET, "testannot")

def rgb2label(img, color_codes = None, one_hot_encode=False):
    if color_codes is None:
        color_codes = {val:i for i,val in enumerate(set( tuple(v) for m2d in img for v in m2d ))}
    n_labels = len(color_codes)
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    result[:,:] = -1
    for rgb, idx in color_codes.items():
        result[(img==rgb).all(2)] = idx

    if one_hot_encode:
        one_hot_labels = np.zeros((img.shape[0],img.shape[1],n_labels))
        # one-hot encoding
        for c in range(n_labels):
            one_hot_labels[: , : , c ] = (result == c ).astype(int)
        result = one_hot_labels

    return result, color_codes

def prepare_directories():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TRAIN_A_DIR, exist_ok=True)
    os.makedirs(VALID_DIR, exist_ok=True)
    os.makedirs(VALID_A_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(TEST_A_DIR, exist_ok=True)

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def copy_to_train():
    png_files = glob.glob("{}/PNGImages/*.png".format(VOC_ROOT))

    for p in png_files:
        img = imread(p)
        imwrite(os.path.join(TRAIN_DIR, os.path.basename(p)), img)

def voc_to_indexed_png():
    mask_files = glob.glob("{}/SegmentationClassPNG/*.png".format(VOC_ROOT))

    colorbgr_codes = {
        (0, 0, 0): 0, # _background_
        (0, 0, 128): 1, # solder
        (0, 128, 0): 2, # crack
        (0, 128, 128): 3 # void
    }

    for m in mask_files:
        img = imread(m)
        img_labels, _ = rgb2label(img, colorbgr_codes)
        imwrite(os.path.join(TRAIN_A_DIR, os.path.basename(m)), img_labels)



def train_valid_test_split(train, valid, test):
    files = glob.glob("%s/*.png" % train[0])
    files.sort()

    i = 0

    for f in files:
        if i % 15 == 0:
            filename = os.path.basename(f)
            shutil.move(f, os.path.join(test[0], filename))
            annot_file = f.replace("train", "trainannot")
            shutil.move(annot_file, os.path.join(test[1], filename))
        elif i % 5 == 0:
            filename = os.path.basename(f)
            shutil.move(f, os.path.join(valid[0], filename))
            annot_file = f.replace("train", "trainannot")
            shutil.move(annot_file, os.path.join(valid[1], filename))

        i = i + 1

def main():
    prepare_directories()
    print("copy original images ...")
    copy_to_train()
    print("convert voc2012 images to indexed png ...")
    voc_to_indexed_png()
    print("train validation test data split ...")
    train_valid_test_split((TRAIN_DIR, TEST_A_DIR), (VALID_DIR, VALID_A_DIR), (TEST_DIR, TEST_A_DIR))

if __name__ == "__main__":
    main()