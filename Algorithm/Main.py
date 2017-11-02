# -*- coding: utf-8 -*-
# Created on Mon Oct 30 2017 9:44:10
# Author: WuLC
# EMail: liangchaowu5@gmail.com

from DeeplearningMethods.AlexNet import AlexNet


if __name__ == '__main__':
    TRAIN_DATA_FILE = './Datasets/D_KDEF_10G_only_rescale_images_with_RBG.pkl'
    TEST_DATA_FILE = './Datasets/D_KDEF_10G_only_rescale_images_with_RBG.pkl'
    FEATURE_FILE_DIR = './Datasets/dl_feature/'
    MODEL_DIR = './models/'
    LOG_DIR = './logs/'
    model = AlexNet(train_data_file = TRAIN_DATA_FILE, 
                    test_data_file=TEST_DATA_FILE,
                    feature_file_dir=FEATURE_FILE_DIR,
                    model_dir=MODEL_DIR,
                    log_dir = LOG_DIR)
    model.build_model()
    model.train_and_eval(dump_feature = False)