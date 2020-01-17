###############################################################
# module: data.py
# description: prepares the image data for ConvNet training
# authors: Vladimir Kulyukin, Prateek Vats, Sarbajit Mukherjee
# bugs to vladimir dot kulyukin at usu dot edu
###############################################################

# grab the image paths and randomly shuffle them
import cv2
import glob
import numpy as np
from sklearn.utils import shuffle
from tflearn import data_utils
import os

# This root directory is on labgforce GPU computer
ROOTDIR_BEE1 = '/home/labgforce/PycharmProjects/PIV_BeeMotions/OmniDirVideo/'

# ROOTDIR_VID  = '/EBM/TEST_VIDS_2019/'
# VID_DIR_R_4_5_may = ROOTDIR_VID + '/R_4_5/may/'
# VID_DIR_R_4_5_jun = ROOTDIR_VID + '/R_4_5/jun/'
# VID_DIR_R_4_5_jul = ROOTDIR_VID + '/R_4_5/jul/'
# VID_DIR_R_4_5_aug = ROOTDIR_VID + '/R_4_5/aug/'

TRAIN_BEE1_BEE = ROOTDIR_BEE1 + 'bee_train/'
TRAIN_BEE1_NO_BEE = ROOTDIR_BEE1 + 'no_bee_train/'

TEST_BEE1_BEE = ROOTDIR_BEE1 + 'bee_test/'
TEST_BEE1_NO_BEE = ROOTDIR_BEE1 + 'no_bee_test/'

VALID_BEE1_BEE = ROOTDIR_BEE1 + 'bee_valid/'
VALID_BEE1_NO_BEE = ROOTDIR_BEE1 + 'no_bee_valid/'

# This root directory is on labgforce GPU computer
# ROOTDIR_BEE2 = 'data/BEE2/'
ROOTDIR_BEE2 = 'data/BEE2'

ROOTDIR_BEE2_1S = ROOTDIR_BEE2 + 'one_super/'
ROOTDIR_BEE2_2S = ROOTDIR_BEE2 + 'two_super/'

TRAIN_1S        = ROOTDIR_BEE2_1S + 'training/'
TRAIN_1S_BEE    = TRAIN_1S + 'bee/'
TRAIN_1S_NO_BEE = TRAIN_1S + 'no_bee/'

TRAIN_2S        = ROOTDIR_BEE2_2S + 'training/'
TRAIN_2S_BEE    = TRAIN_2S + 'bee/'
TRAIN_2S_NO_BEE = TRAIN_2S + 'no_bee/'

TEST_1S        = ROOTDIR_BEE2_1S + 'testing/'
TEST_1S_BEE    = TEST_1S + 'bee/'
TEST_1S_NO_BEE = TEST_1S + 'no_bee/'

TEST_2S        = ROOTDIR_BEE2_2S + 'testing/'
TEST_2S_BEE    = TEST_2S + 'bee/'
TEST_2S_NO_BEE = TEST_2S + 'no_bee/'

VALID_1S        = ROOTDIR_BEE2_1S + 'validation/'
VALID_1S_BEE    = VALID_1S + 'bee/'
VALID_1S_NO_BEE = VALID_1S + 'no_bee/'

VALID_2S        = ROOTDIR_BEE2_2S + 'validation/'
VALID_2S_BEE    = VALID_2S + 'bee/'
VALID_2S_NO_BEE = VALID_2S + 'no_bee/'

BEE    = 0
NO_BEE = 1

BEE1_IMWIDTH = 32
BEE1_IMHEIGHT = 32
BEE1_IMCHANNEL = 3

BEE2_IMWIDTH   = 64
BEE2_IMHEIGHT  = 64
BEE2_IMCHANNEL = 3


# Directories where trained models are presisted on my GPU computer.
PERSIST_BEE1_DIR = '/home/labgforce/PycharmProjects/PIV_BeeMotions/persisted_bee1_models/'
PERSIST_BEE2_DIR = '/home/labgforce/PycharmProjects/PIV_BeeMotions/persisted_bee2_models/'

def generate_file_names(ftype, rootdir):
    """
    :param ftype: file type (e.g., '.png')
    :param rootdir: a directory from where the walk starts.
    :return:

    recursively walk dir tree beginning from rootdir
    and generate full paths to all files that end with ftype.
    sample call: generate_file_names('.jpg', /home/pi/images/')
    '''
    """

    for path, dirlist, filelist in os.walk(rootdir):
        for file_name in filelist:
            if not file_name.startswith('.') and \
               file_name.endswith(ftype):
                yield os.path.join(path, file_name)
        for d in dirlist:
            generate_file_names(ftype, d)

def get_image_paths(rootdir, ftype='.png'):
    """
    returns a list of image paths to all images in rootdir, recursively walks all
    subdirectories in rootdir if those exist.
    :param ftype: file type (e.g., '.png')
    :param rootdir: a directory
    :return: a list of file paths.
    """
    return [p for p in generate_file_names(ftype, rootdir)]

### ========================= BEE1 DATA =========================================

def get_train_bee1_bee_image_paths(ftype='.png'):
    global TRAIN_BEE1_BEE
    return get_image_paths(TRAIN_BEE1_BEE, ftype=ftype)

def label_train_bee1_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_train_bee1_bee_image_paths(ftype=ftype)]

def get_train_bee1_no_bee_image_paths(ftype='.png'):
    global TRAIN_BEE1_NO_BEE
    return get_image_paths(TRAIN_BEE1_NO_BEE, ftype=ftype)

def label_train_bee1_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_train_bee1_no_bee_image_paths(ftype=ftype)]

def get_test_bee1_bee_image_paths(ftype='.png'):
    global TEST_BEE1_BEE
    return get_image_paths(TEST_BEE1_BEE, ftype=ftype)

def label_test_bee1_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_test_bee1_bee_image_paths(ftype=ftype)]

def get_test_bee1_no_bee_image_paths(ftype='.png'):
    global TEST_BEE1_NO_BEE
    return get_image_paths(TEST_BEE1_NO_BEE, ftype=ftype)

def label_test_bee1_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_test_bee1_no_bee_image_paths(ftype=ftype)]

def get_valid_bee1_bee_image_paths(ftype='.png'):
    global VALID_BEE1_BEE
    return get_image_paths(VALID_BEE1_BEE, ftype=ftype)

def label_valid_bee1_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_valid_bee1_bee_image_paths(ftype=ftype)]

def get_valid_bee1_no_bee_image_paths(ftype='.png'):
    global VALID_BEE1_NO_BEE
    return get_image_paths(VALID_BEE1_NO_BEE, ftype=ftype)

def label_valid_bee1_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_valid_bee1_no_bee_image_paths(ftype=ftype)]    

### ========================= BEE2 DATA =========================================

def get_train_bee2_1s_bee_image_paths(ftype='.png'):
    global TRAIN_1S_BEE
    return get_image_paths(TRAIN_1S_BEE, ftype=ftype)

def label_train_bee2_1s_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_train_bee2_1s_bee_image_paths(ftype=ftype)]

def get_train_bee2_1s_no_bee_image_paths(ftype='.png'):
    global TRAIN_1S_NO_BEE
    return get_image_paths(TRAIN_1S_NO_BEE, ftype=ftype)

def label_train_bee2_1s_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_train_bee2_1s_no_bee_image_paths(ftype=ftype)]

def get_test_bee2_1s_bee_image_paths(ftype='.png'):
    global TEST_1S_BEE
    return get_image_paths(TEST_1S_BEE, ftype=ftype)

def label_test_bee2_1s_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_test_bee2_1s_bee_image_paths(ftype=ftype)]

def get_test_bee2_1s_no_bee_image_paths(ftype='.png'):
    global TEST_1S_NO_BEE
    return get_image_paths(TEST_1S_NO_BEE, ftype=ftype)

def label_test_bee2_1s_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_test_bee2_1s_no_bee_image_paths(ftype=ftype)]

def get_valid_bee2_1s_bee_image_paths(ftype='.png'):
    global VALID_1S_BEE
    return get_image_paths(VALID_1S_BEE, ftype=ftype)

def label_valid_bee2_1s_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_valid_bee2_1s_bee_image_paths(ftype=ftype)]

def get_valid_bee2_1s_no_bee_image_paths(ftype='.png'):
    global VALID_1S_NO_BEE
    return get_image_paths(VALID_1S_NO_BEE, ftype=ftype)

def label_valid_bee2_1s_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_valid_bee2_1s_no_bee_image_paths(ftype=ftype)]

### 2-super image data paths
def get_train_bee2_2s_bee_image_paths(ftype='.png'):
    global TRAIN_2S_BEE
    return get_image_paths(TRAIN_2S_BEE, ftype=ftype)

def label_train_bee2_2s_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_train_bee2_2s_bee_image_paths(ftype=ftype)]

def get_train_bee2_2s_no_bee_image_paths(ftype='.png'):
    global TRAIN_2S_NO_BEE
    return get_image_paths(TRAIN_2S_NO_BEE, ftype=ftype)

def label_train_bee2_2s_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_train_bee2_2s_no_bee_image_paths(ftype=ftype)]

def get_test_bee2_2s_bee_image_paths(ftype='.png'):
    global TEST_2S_BEE
    return get_image_paths(TEST_2S_BEE, ftype=ftype)

def label_test_bee2_2s_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_test_bee2_2s_bee_image_paths(ftype=ftype)]

def get_test_bee2_2s_no_bee_image_paths(ftype='.png'):
    global TEST_2S_NO_BEE
    return get_image_paths(TEST_2S_NO_BEE, ftype=ftype)

def label_test_bee2_2s_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_test_bee2_2s_no_bee_image_paths(ftype=ftype)]

def get_valid_bee2_2s_bee_image_paths(ftype='.png'):
    global VALID_2S_BEE
    return get_image_paths(VALID_2S_BEE, ftype=ftype)

def label_valid_bee2_2s_bee_image_paths(ftype='.png'):
    global BEE
    return [(p, BEE) for p in get_valid_bee2_2s_bee_image_paths(ftype=ftype)]

def get_valid_bee2_2s_no_bee_image_paths(ftype='.png'):
    global VALID_2S_NO_BEE
    return get_image_paths(VALID_2S_NO_BEE, ftype=ftype)

def label_valid_bee2_2s_no_bee_image_paths(ftype='.png'):
    global NO_BEE
    return [(p, NO_BEE) for p in get_valid_bee2_2s_no_bee_image_paths(ftype=ftype)]

def get_labeled_image_paths(dataset='', data_shuffle=True, ftype='.png'):
    paths = None
    if dataset == 'train_bee1_bee':
        paths = label_train_bee1_bee_image_paths(ftype=ftype)
    elif dataset == 'train_bee1_no_bee':
        paths = label_train_bee1_no_bee_image_paths(ftype=ftype)
    elif dataset == 'test_bee1_bee':
        paths = label_test_bee1_bee_image_paths(ftype=ftype)
    elif dataset == 'test_bee1_no_bee':
        paths = label_test_bee1_no_bee_image_paths(ftype=ftype)
    elif dataset == 'valid_bee1_bee':
        paths = label_valid_bee1_bee_image_paths(ftype=ftype)
    elif dataset == 'valid_bee1_no_bee':
        paths = label_valid_bee1_no_bee_image_paths(ftype=ftype)
    elif dataset == 'train_bee2_1s_bee':
        paths = label_train_bee2_1s_bee_image_paths(ftype=ftype)
    elif dataset == 'train_bee2_1s_no_bee':
        paths = label_train_bee2_1s_no_bee_image_paths(ftype=ftype)
    elif dataset == 'test_bee2_1s_bee':
        paths = label_test_bee2_1s_bee_image_paths(ftype=ftype)
    elif dataset == 'test_bee2_1s_no_bee':
        paths = label_test_bee2_1s_no_bee_image_paths(ftype=ftype)
    elif dataset == 'valid_bee2_1s_bee':
        paths = label_valid_bee2_1s_bee_image_paths(ftype=ftype)
    elif dataset == 'valid_bee2_1s_no_bee':
        paths = label_valid_bee2_1s_no_bee_image_paths(ftype=ftype)
    elif dataset == 'train_bee2_2s_bee':
        paths = label_train_bee2_2s_bee_image_paths(ftype=ftype)
    elif dataset == 'train_bee2_2s_no_bee':
        paths = label_train_bee2_2s_no_bee_image_paths(ftype=ftype)
    elif dataset == 'test_bee2_2s_bee':
        paths = label_test_bee2_2s_bee_image_paths(ftype=ftype)
    elif dataset == 'test_bee2_2s_no_bee':
        paths = label_test_bee2_2s_no_bee_image_paths(ftype=ftype)
    elif dataset == 'valid_bee2_2s_bee':
        paths = label_valid_bee2_2s_bee_image_paths(ftype=ftype)
    elif dataset == 'valid_bee2_2s_no_bee':
        paths = label_valid_bee2_2s_no_bee_image_paths(ftype=ftype)
    else:
        raise Exception('invalid dataset: ' + str(dataset))

    if data_shuffle == True:
        return shuffle(paths)
    else:
        return paths

def prepare_data(labeled_image_paths, image_width=64, image_height=64, start_index=0, end_index=0):
    """
    takes image paths of labeled images (labeled_image_paths) returned from get_raw_image_paths() and 
    converts them into numpy arrays batchX and batchY for ConvNet training; images are normalized.
    :param image_paths:
    :param image_width:
    :param image_height:
    :param start_index:
    :param end_index:
    :return: numpy arrays batchX and batchY for ConvNet training.
    """
    batchX = []
    batchY = []

    if end_index == 0:
        end_index = len(labeled_image_paths)

    for i in range(start_index, end_index):
        image = cv2.imread(labeled_image_paths[i][0])
        if image.shape[0] > 0 and image.shape[1] > 0:
            image = cv2.resize(image, (image_width, image_height))
            image = np.array(image)[:, :, 0:3]
            batchX.append(image)
            batchY.append(labeled_image_paths[i][1])

    batchX = np.array(batchX, dtype='float32') / 255.0
    batchY = np.array(batchY)
    #print('Old batchY={}'.format(batchY))
    batchY = data_utils.to_categorical(batchY, nb_classes=2)
    # batchY has the form [[1. 0.], [1. 0.],..., [0. 1.], [0. 1.]],
    # where [1, 0] stands for BEE and [0, 1] for NOBEE.
    #print('New batchY={}'.format(batchY))
    return batchX, batchY

def unit_test_01():
    train_image_paths = get_labeled_image_paths(dataset='train_bee2_1s_bee', ftype='.png')[:5]
    train_image_paths += get_labeled_image_paths(dataset='train_bee2_1s_no_bee', ftype='.png')[:5]
    for p in train_image_paths:
        print(p)
    batchX, batchY = prepare_data(train_image_paths, image_width=BEE2_IMWIDTH, image_height=BEE2_IMHEIGHT)
    for p, b in zip(train_image_paths, zip(batchX, batchY)):
        print('{} -> {}, {}'.format(p, b[0].shape, b[1]))


if __name__ == '__main__':
    unit_test_01()



