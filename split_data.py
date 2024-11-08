import random
import os
import numpy as np
import shutil


if __name__ == "__main__":
    modality_name_list = {'t1': '_t1.nii.gz', 
                        't1ce': '_t1ce.nii.gz', 
                        't2': '_t2.nii.gz', 
                        'flair': '_flair.nii.gz'}
    src_dir_HGG = "/data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/HGG"
    src_dir_LGG = '/data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/LGG'
    train_dir = "/data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/train"
    test_dir = "/data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/test"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    # src_list = []
    # src_list.extend([os.path.join(src_dir_HGG, item) for item in os.listdir(src_dir_HGG)] )
    # src_list.extend([os.path.join(src_dir_LGG, item) for item in os.listdir(src_dir_LGG)] )
    # random.shuffle(src_list)
    # spilt_point = int(0.8 * len(src_list))
    # train_list = src_list[:spilt_point]
    # test_list = src_list[spilt_point:]

    train_list = []
    test_list = []
    with open("Brats_train.list", "r") as file:
        for line in file:
            tmp = os.path.join(src_dir_HGG, line.strip())
            if not os.path.exists(tmp):
                tmp = os.path.join(src_dir_LGG, line.strip())
            train_list.append(tmp)
    with open("Brats_test.list", "r") as file:
        for line in file:
            tmp = os.path.join(src_dir_HGG, line.strip())
            if not os.path.exists(tmp):
                tmp = os.path.join(src_dir_LGG, line.strip())
            test_list.append(tmp)

    for item in train_list:
        print(item)
        dest_path = os.path.join(train_dir, os.path.basename(item))
        shutil.copytree(item, dest_path)
    
    for item in test_list:
        dest_path = os.path.join(test_dir, os.path.basename(item))
        shutil.copytree(item, dest_path)