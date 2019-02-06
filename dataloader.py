#coding:utf-8
import os
from os import listdir
from os.path import join

from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # for test
from torchvision.utils import save_image, make_grid


"""
train.py もしくは test.py にて,
###
from dataloader import get_training_set, get_test_set
###
                                  #############
で import した後，このスクリプトの一番下の ## testing ## のように書くことで呼び出せる
                                  #############
入力画像 => GroundTruth のように学習する場合
## Input Image ##
"input_image"
## GroundTruth Image ##
"gt_img"
    root_path/  --- /train/ --- /input_img/
                 |           └- /mask_img/
                 |           └- /gt_img/
                 |
                 └- /val/ ---- /input_img/
                 |          └- /mask_img/
                 |          └- /gt_img/
                 |
                 └- /test/ --- /input_img/
                            └- /mask_img/

"""

class DatasetFromFolder(data.Dataset):
    def __init__(self, data_directory):
        super(DatasetFromFolder, self).__init__()
        self.input_img_path = join(data_directory, 'input_img')
        self.mask_img_path = join(data_directory, 'mask_img')
        self.gt_img_path = join(data_directory, 'gt_img')

        self.input_img_filenames = [x for x in listdir(self.input_img_path) if is_image_file(x)]
        self.input_img_filenames.sort()
        self.mask_img_filenames = [x for x in listdir(self.mask_img_path) if is_image_file(x)]
        self.mask_img_filenames.sort()
        self.gt_img_filenames = [x for x in listdir(self.gt_img_path) if is_image_file(x)]
        self.gt_img_filenames.sort()

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        mask_tf_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)
        self.mask_tf = transforms.Compose(mask_tf_list)

    def __len__(self):
        return len(self.input_img_filenames)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.input_img_path, self.input_img_filenames[index]))
        input = self.transform(input)
        mask = load_img(join(self.mask_img_path, self.mask_img_filenames[index]))
        mask = self.mask_tf(mask)
        target = load_img(join(self.gt_img_path, self.gt_img_filenames[index]))
        target = self.transform(target)
        return input, mask, target

# class ValDatasetFromFolder(data.Dataset):
#     def __init__(self, data_directory):
#         super(ValDatasetFromFolder, self).__init__()
#         self.input_img_path = join(data_directory, 'input_img')
#
#         self.input_img_filenames = [x for x in listdir(self.input_img_path) if is_image_file(x)]
#         self.input_img_filenames.sort()
#
#         transform_list = [transforms.ToTensor(),
#                           ]
#
#         self.transform = transforms.Compose(transform_list)
#
#     def __len__(self):
#         return len(self.input_img_filenames)
#
#     def __getitem__(self, index):
#         # Load Image
#         input = load_img(join(self.input_img_path, self.input_img_filenames[index]))
#         input = self.transform(input)
#         return input


def get_training_set(root_dir):
    train_dir = join(root_dir, 'train')
    return DatasetFromFolder(train_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, 'val')
    return DatasetFromFolder(val_dir)

def get_test_set(root_dir):
    test_dir = join(root_dir, 'test')
    return DatasetFromFolder(test_dir)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img

if __name__ == '__main__':
    ###########
    # testing #
    ###########

    print('==> Preparing Data Set')
    root_path = './dataset/'
    train_set = get_training_set(root_path)
    # val_set = get_val_set(root_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=1,
                                        batch_size=4, shuffle=True)
    # val_data_loader = DataLoader(dataset=val_set, num_workers=1,
                                        # batch_size=4, shuffle=False)
    print('==> Preparing Data Set: Complete\n')

    #################################
    ######### visualization #########
    #################################

    for iteration, batch in enumerate(training_data_loader):
        img, mask, gt = batch[0], batch[1], batch[2]

        # input_images = 'inputs.png'
        # input_image = 'input.png'
        # gt_images = 'gts.png'
        # gt_image = 'gt.png'

        testDataloaderDir = './test/DataLoader/'
        os.makedirs(testDataloaderDir, exist_ok=True)
        imgList = [ img[0], mask[0], gt[0] ]
        grid_img = make_grid(imgList)
        save_image(grid_img, testDataloaderDir + '{}'.format('images.png'))
        # save_image(b.data, testDataloaderDir + '{}'.format(gt_images))
        # save_image(a.data[1], testDataloaderDir + '{}'.format(input_image))
        # save_image(b.data[1], testDataloaderDir + '{}'.format(gt_image))
        if(iteration == 0): break
