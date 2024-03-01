import h5py
import os
import json
import pickle
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


class flintstones_helper(object):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self):
        self.data_dir = '../data/flintstones_data'

        splits = json.load(open(os.path.join(self.data_dir, 'train-val-test_split.json'), 'r'))
        self.train_ids, self.val_ids, self.test_ids = splits["train"], splits["val"], splits["test"]
        self.followings = pickle.load(open(os.path.join(self.data_dir, 'following_cache4.pkl'), 'rb'))
        self._followings = self.followings.copy()
        self.annotations = json.load(open(os.path.join(self.data_dir, 'flintstones_annotations_v1-0.json')))
        self.nominal_name_mapping = json.load(open(os.path.join(self.data_dir, 'char_name_mapping.json'), 'r'))
        self.unseen_char_anno = json.load(open(os.path.join(self.data_dir, 'flintstones_unseen_anno.json'), 'r'))
        self.h5file = h5py.File("/home/xiyu/projects/AR-LDM/data/flintstones_rare-char_rmeoved.h5", "r")
        self.seen_len = {"train": len(self.h5file['train']['text']), "test": len(self.h5file['test']['text'])}

        self.descriptions = dict()
        for sample in self.annotations:
            self.descriptions[sample["globalID"]] = sample["description"]

        pass

    def get_annotations(self, id):
        return self.descriptions[id]

    def show_images(self, id):
        # load the npy file as pil images
        img = os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(id))
        img = np.load(img)[0]
        img = Image.fromarray(img)
        # plot
        plt.imshow(img)
        plt.show()

    def save_images(self, path, id):
        img = os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(id))
        img = np.load(img)[0]
        img = Image.fromarray(img)
        img.save(os.path.join(path, '{}.png'.format(id)))

    def save_description(self):
        # save the descriptions
        with open(os.path.join(self.data_dir, 'flintstones_descriptions.txt'), 'w') as f:
            for key, value in self.descriptions.items():
                f.write('{}: {}\n'.format(key, value))

if __name__ == "__main__":
    helper = flintstones_helper()
    helper.save_description()
    # _tmp = {'s_02_e_26_shot_009480_009554': ['s_02_e_26_shot_009480_009554', 's_02_e_26_shot_009757_009831', 's_02_e_26_shot_010219_010293', 's_02_e_26_shot_010294_010368', 's_02_e_26_shot_010538_010612'], 's_02_e_26_shot_009757_009831': ['s_02_e_26_shot_009757_009831', 's_02_e_26_shot_010219_010293', 's_02_e_26_shot_010294_010368', 's_02_e_26_shot_010538_010612', 's_02_e_26_shot_010813_010887'], 's_02_e_26_shot_010219_010293': ['s_02_e_26_shot_010219_010293', 's_02_e_26_shot_010294_010368', 's_02_e_26_shot_010538_010612', 's_02_e_26_shot_010813_010887', 's_02_e_26_shot_010888_010962'], 's_02_e_26_shot_010294_010368': ['s_02_e_26_shot_010294_010368', 's_02_e_26_shot_010538_010612', 's_02_e_26_shot_010813_010887', 's_02_e_26_shot_010888_010962', 's_02_e_26_shot_010989_011063'], 's_02_e_26_shot_010538_010612': ['s_02_e_26_shot_010538_010612', 's_02_e_26_shot_010813_010887', 's_02_e_26_shot_010888_010962', 's_02_e_26_shot_010989_011063', 's_02_e_26_shot_011064_011138'], 's_02_e_26_shot_010813_010887': ['s_02_e_26_shot_010813_010887', 's_02_e_26_shot_010888_010962', 's_02_e_26_shot_010989_011063', 's_02_e_26_shot_011064_011138', 's_02_e_26_shot_011231_011305'], 's_02_e_26_shot_010888_010962': ['s_02_e_26_shot_010888_010962', 's_02_e_26_shot_010989_011063', 's_02_e_26_shot_011064_011138', 's_02_e_26_shot_011231_011305', 's_02_e_26_shot_011627_011701'], 's_02_e_26_shot_010989_011063': ['s_02_e_26_shot_010989_011063', 's_02_e_26_shot_011064_011138', 's_02_e_26_shot_011231_011305', 's_02_e_26_shot_011627_011701', 's_02_e_26_shot_011726_011800'], 's_02_e_26_shot_013638_013712': ['s_02_e_26_shot_013638_013712', 's_02_e_26_shot_013713_013787', 's_02_e_26_shot_013788_013862', 's_02_e_26_shot_013863_013937', 's_02_e_26_shot_014223_014297'], 's_02_e_26_shot_013713_013787': ['s_02_e_26_shot_013713_013787', 's_02_e_26_shot_013788_013862', 's_02_e_26_shot_013863_013937', 's_02_e_26_shot_014223_014297', 's_02_e_26_shot_014432_014506'], 's_02_e_26_shot_013788_013862': ['s_02_e_26_shot_013788_013862', 's_02_e_26_shot_013863_013937', 's_02_e_26_shot_014223_014297', 's_02_e_26_shot_014432_014506', 's_02_e_26_shot_014883_014957'], 's_02_e_26_shot_013863_013937': ['s_02_e_26_shot_013863_013937', 's_02_e_26_shot_014223_014297', 's_02_e_26_shot_014432_014506', 's_02_e_26_shot_014883_014957', 's_02_e_26_shot_014958_015032'], 's_02_e_26_shot_023639_023713': ['s_02_e_26_shot_023639_023713', 's_02_e_26_shot_023892_023966', 's_02_e_26_shot_023967_024041', 's_02_e_26_shot_024343_024417', 's_02_e_26_shot_024583_024657'], 's_02_e_26_shot_023892_023966': ['s_02_e_26_shot_023892_023966', 's_02_e_26_shot_023967_024041', 's_02_e_26_shot_024343_024417', 's_02_e_26_shot_024583_024657', 's_02_e_26_shot_024658_024732'], 's_02_e_26_shot_023967_024041': ['s_02_e_26_shot_023967_024041', 's_02_e_26_shot_024343_024417', 's_02_e_26_shot_024583_024657', 's_02_e_26_shot_024658_024732', 's_02_e_26_shot_024883_024957'], 's_02_e_26_shot_024343_024417': ['s_02_e_26_shot_024343_024417', 's_02_e_26_shot_024583_024657', 's_02_e_26_shot_024658_024732', 's_02_e_26_shot_024883_024957', 's_02_e_26_shot_025025_025099'], 's_02_e_26_shot_024583_024657': ['s_02_e_26_shot_024583_024657', 's_02_e_26_shot_024658_024732', 's_02_e_26_shot_024883_024957', 's_02_e_26_shot_025025_025099', 's_02_e_26_shot_025100_025174'], 's_02_e_26_shot_024658_024732': ['s_02_e_26_shot_024658_024732', 's_02_e_26_shot_024883_024957', 's_02_e_26_shot_025025_025099', 's_02_e_26_shot_025100_025174', 's_02_e_26_shot_025175_025249'], 's_02_e_26_shot_024883_024957': ['s_02_e_26_shot_024883_024957', 's_02_e_26_shot_025025_025099', 's_02_e_26_shot_025100_025174', 's_02_e_26_shot_025175_025249', 's_02_e_26_shot_025443_025517'], 's_02_e_26_shot_032813_032887': ['s_02_e_26_shot_032813_032887', 's_02_e_26_shot_032888_032962', 's_02_e_26_shot_033188_033262', 's_02_e_26_shot_034144_034218', 's_02_e_26_shot_034329_034403'], 's_02_e_26_shot_032888_032962': ['s_02_e_26_shot_032888_032962', 's_02_e_26_shot_033188_033262', 's_02_e_26_shot_034144_034218', 's_02_e_26_shot_034329_034403', 's_02_e_26_shot_034661_034735'], 's_02_e_26_shot_033188_033262': ['s_02_e_26_shot_033188_033262', 's_02_e_26_shot_034144_034218', 's_02_e_26_shot_034329_034403', 's_02_e_26_shot_034661_034735', 's_02_e_26_shot_034736_034810'], 's_02_e_26_shot_034144_034218': ['s_02_e_26_shot_034144_034218', 's_02_e_26_shot_034329_034403', 's_02_e_26_shot_034661_034735', 's_02_e_26_shot_034736_034810', 's_02_e_26_shot_035552_035626'], 's_02_e_26_shot_034329_034403': ['s_02_e_26_shot_034329_034403', 's_02_e_26_shot_034661_034735', 's_02_e_26_shot_034736_034810', 's_02_e_26_shot_035552_035626', 's_02_e_26_shot_035702_035776'], 's_02_e_26_shot_034661_034735': ['s_02_e_26_shot_034661_034735', 's_02_e_26_shot_034736_034810', 's_02_e_26_shot_035552_035626', 's_02_e_26_shot_035702_035776', 's_02_e_26_shot_035777_035851'], 's_02_e_26_shot_034736_034810': ['s_02_e_26_shot_034736_034810', 's_02_e_26_shot_035552_035626', 's_02_e_26_shot_035702_035776', 's_02_e_26_shot_035777_035851', 's_02_e_26_shot_036309_036383'], 's_02_e_26_shot_038027_038101': ['s_02_e_26_shot_038027_038101', 's_02_e_26_shot_038214_038288', 's_02_e_26_shot_038542_038616', 's_02_e_26_shot_039160_039234', 's_02_e_26_shot_039851_039925'], 's_02_e_26_shot_038214_038288': ['s_02_e_26_shot_038214_038288', 's_02_e_26_shot_038542_038616', 's_02_e_26_shot_039160_039234', 's_02_e_26_shot_039851_039925', 's_02_e_26_shot_039926_040000'], 's_02_e_26_shot_038542_038616': ['s_02_e_26_shot_038542_038616', 's_02_e_26_shot_039160_039234', 's_02_e_26_shot_039851_039925', 's_02_e_26_shot_039926_040000', 's_02_e_26_shot_040076_040150'], 's_02_e_26_shot_039160_039234': ['s_02_e_26_shot_039160_039234', 's_02_e_26_shot_039851_039925', 's_02_e_26_shot_039926_040000', 's_02_e_26_shot_040076_040150', 's_02_e_26_shot_040348_040422'], 's_02_e_26_shot_041169_041243': ['s_02_e_26_shot_041169_041243', 's_02_e_26_shot_041244_041318', 's_02_e_26_shot_041514_041588', 's_02_e_26_shot_041820_041894', 's_02_e_26_shot_042075_042149'], 's_02_e_26_shot_041244_041318': ['s_02_e_26_shot_041244_041318', 's_02_e_26_shot_041514_041588', 's_02_e_26_shot_041820_041894', 's_02_e_26_shot_042075_042149', 's_02_e_26_shot_042251_042325'], 's_02_e_26_shot_041514_041588': ['s_02_e_26_shot_041514_041588', 's_02_e_26_shot_041820_041894', 's_02_e_26_shot_042075_042149', 's_02_e_26_shot_042251_042325', 's_02_e_26_shot_042339_042413'], 's_02_e_26_shot_041820_041894': ['s_02_e_26_shot_041820_041894', 's_02_e_26_shot_042075_042149', 's_02_e_26_shot_042251_042325', 's_02_e_26_shot_042339_042413', 's_02_e_26_shot_042812_042886'], 's_02_e_26_shot_042075_042149': ['s_02_e_26_shot_042075_042149', 's_02_e_26_shot_042251_042325', 's_02_e_26_shot_042339_042413', 's_02_e_26_shot_042812_042886', 's_02_e_26_shot_042887_042961'], 's_02_e_26_shot_042251_042325': ['s_02_e_26_shot_042251_042325', 's_02_e_26_shot_042339_042413', 's_02_e_26_shot_042812_042886', 's_02_e_26_shot_042887_042961', 's_02_e_26_shot_042999_043073'], 's_02_e_26_shot_042339_042413': ['s_02_e_26_shot_042339_042413', 's_02_e_26_shot_042812_042886', 's_02_e_26_shot_042887_042961', 's_02_e_26_shot_042999_043073', 's_02_e_26_shot_043074_043148'], 's_02_e_26_shot_043224_043298': ['s_02_e_26_shot_043224_043298', 's_02_e_26_shot_043615_043689', 's_02_e_26_shot_044143_044217', 's_02_e_26_shot_044218_044292', 's_02_e_26_shot_044352_044426'], 's_02_e_26_shot_043615_043689': ['s_02_e_26_shot_043615_043689', 's_02_e_26_shot_044143_044217', 's_02_e_26_shot_044218_044292', 's_02_e_26_shot_044352_044426', 's_02_e_26_shot_044814_044888'], 's_02_e_26_shot_044143_044217': ['s_02_e_26_shot_044143_044217', 's_02_e_26_shot_044218_044292', 's_02_e_26_shot_044352_044426', 's_02_e_26_shot_044814_044888', 's_02_e_26_shot_045296_045370'], 's_03_e_08_shot_027962_028036': ['s_03_e_08_shot_027962_028036', 's_03_e_08_shot_028226_028300', 's_03_e_08_shot_028842_028916', 's_03_e_08_shot_028917_028991', 's_03_e_08_shot_029159_029233'], 's_03_e_08_shot_028226_028300': ['s_03_e_08_shot_028226_028300', 's_03_e_08_shot_028842_028916', 's_03_e_08_shot_028917_028991', 's_03_e_08_shot_029159_029233', 's_03_e_08_shot_029434_029508'], 's_03_e_08_shot_028842_028916': ['s_03_e_08_shot_028842_028916', 's_03_e_08_shot_028917_028991', 's_03_e_08_shot_029159_029233', 's_03_e_08_shot_029434_029508', 's_03_e_08_shot_029656_029730'], 's_03_e_08_shot_028917_028991': ['s_03_e_08_shot_028917_028991', 's_03_e_08_shot_029159_029233', 's_03_e_08_shot_029434_029508', 's_03_e_08_shot_029656_029730', 's_03_e_08_shot_029907_029981'], 's_03_e_08_shot_029159_029233': ['s_03_e_08_shot_029159_029233', 's_03_e_08_shot_029434_029508', 's_03_e_08_shot_029656_029730', 's_03_e_08_shot_029907_029981', 's_03_e_08_shot_030127_030201'], 's_03_e_08_shot_029434_029508': ['s_03_e_08_shot_029434_029508', 's_03_e_08_shot_029656_029730', 's_03_e_08_shot_029907_029981', 's_03_e_08_shot_030127_030201', 's_03_e_08_shot_030367_030441'], 's_03_e_08_shot_029656_029730': ['s_03_e_08_shot_029656_029730', 's_03_e_08_shot_029907_029981', 's_03_e_08_shot_030127_030201', 's_03_e_08_shot_030367_030441', 's_03_e_08_shot_030459_030533'], 's_03_e_08_shot_029907_029981': ['s_03_e_08_shot_029907_029981', 's_03_e_08_shot_030127_030201', 's_03_e_08_shot_030367_030441', 's_03_e_08_shot_030459_030533', 's_03_e_08_shot_030899_030973'], 's_03_e_08_shot_030127_030201': ['s_03_e_08_shot_030127_030201', 's_03_e_08_shot_030367_030441', 's_03_e_08_shot_030459_030533', 's_03_e_08_shot_030899_030973', 's_03_e_08_shot_031124_031198'], 's_03_e_08_shot_030367_030441': ['s_03_e_08_shot_030367_030441', 's_03_e_08_shot_030459_030533', 's_03_e_08_shot_030899_030973', 's_03_e_08_shot_031124_031198', 's_03_e_08_shot_031381_031455'], 's_03_e_08_shot_030459_030533': ['s_03_e_08_shot_030459_030533', 's_03_e_08_shot_030899_030973', 's_03_e_08_shot_031124_031198', 's_03_e_08_shot_031381_031455', 's_03_e_08_shot_031681_031755'], 's_03_e_08_shot_030899_030973': ['s_03_e_08_shot_030899_030973', 's_03_e_08_shot_031124_031198', 's_03_e_08_shot_031381_031455', 's_03_e_08_shot_031681_031755', 's_03_e_08_shot_031834_031908'], 's_03_e_08_shot_031124_031198': ['s_03_e_08_shot_031124_031198', 's_03_e_08_shot_031381_031455', 's_03_e_08_shot_031681_031755', 's_03_e_08_shot_031834_031908', 's_03_e_08_shot_032338_032412'], 's_03_e_08_shot_031381_031455': ['s_03_e_08_shot_031381_031455', 's_03_e_08_shot_031681_031755', 's_03_e_08_shot_031834_031908', 's_03_e_08_shot_032338_032412', 's_03_e_08_shot_032516_032590'], 's_03_e_20_shot_000792_000866': ['s_03_e_20_shot_000792_000866', 's_03_e_20_shot_000902_000976', 's_03_e_20_shot_000977_001051', 's_03_e_20_shot_002607_002681', 's_03_e_20_shot_002682_002756'], 's_03_e_20_shot_011418_011492': ['s_03_e_20_shot_011418_011492', 's_03_e_20_shot_011572_011646', 's_03_e_20_shot_011979_012053', 's_03_e_20_shot_012054_012128', 's_03_e_20_shot_012230_012304'], 's_03_e_20_shot_011572_011646': ['s_03_e_20_shot_011572_011646', 's_03_e_20_shot_011979_012053', 's_03_e_20_shot_012054_012128', 's_03_e_20_shot_012230_012304', 's_03_e_20_shot_012518_012592'], 's_03_e_20_shot_011979_012053': ['s_03_e_20_shot_011979_012053', 's_03_e_20_shot_012054_012128', 's_03_e_20_shot_012230_012304', 's_03_e_20_shot_012518_012592', 's_03_e_20_shot_012650_012724'], 's_03_e_20_shot_012054_012128': ['s_03_e_20_shot_012054_012128', 's_03_e_20_shot_012230_012304', 's_03_e_20_shot_012518_012592', 's_03_e_20_shot_012650_012724', 's_03_e_20_shot_012804_012878'], 's_03_e_20_shot_012230_012304': ['s_03_e_20_shot_012230_012304', 's_03_e_20_shot_012518_012592', 's_03_e_20_shot_012650_012724', 's_03_e_20_shot_012804_012878', 's_03_e_20_shot_014263_014337'], 's_03_e_20_shot_012518_012592': ['s_03_e_20_shot_012518_012592', 's_03_e_20_shot_012650_012724', 's_03_e_20_shot_012804_012878', 's_03_e_20_shot_014263_014337', 's_03_e_20_shot_014509_014583'], 's_03_e_20_shot_012650_012724': ['s_03_e_20_shot_012650_012724', 's_03_e_20_shot_012804_012878', 's_03_e_20_shot_014263_014337', 's_03_e_20_shot_014509_014583', 's_03_e_20_shot_015510_015584'], 's_03_e_20_shot_012804_012878': ['s_03_e_20_shot_012804_012878', 's_03_e_20_shot_014263_014337', 's_03_e_20_shot_014509_014583', 's_03_e_20_shot_015510_015584', 's_03_e_20_shot_015948_016022'], 's_03_e_20_shot_014263_014337': ['s_03_e_20_shot_014263_014337', 's_03_e_20_shot_014509_014583', 's_03_e_20_shot_015510_015584', 's_03_e_20_shot_015948_016022', 's_03_e_20_shot_016357_016431'], 's_03_e_20_shot_014509_014583': ['s_03_e_20_shot_014509_014583', 's_03_e_20_shot_015510_015584', 's_03_e_20_shot_015948_016022', 's_03_e_20_shot_016357_016431', 's_03_e_20_shot_016654_016728'], 's_03_e_20_shot_016357_016431': ['s_03_e_20_shot_016357_016431', 's_03_e_20_shot_016654_016728', 's_03_e_20_shot_017215_017289', 's_03_e_20_shot_017358_017432', 's_03_e_20_shot_017785_017859'], 's_03_e_20_shot_016654_016728': ['s_03_e_20_shot_016654_016728', 's_03_e_20_shot_017215_017289', 's_03_e_20_shot_017358_017432', 's_03_e_20_shot_017785_017859', 's_03_e_20_shot_018192_018266'], 's_03_e_20_shot_017215_017289': ['s_03_e_20_shot_017215_017289', 's_03_e_20_shot_017358_017432', 's_03_e_20_shot_017785_017859', 's_03_e_20_shot_018192_018266', 's_03_e_20_shot_018304_018378'], 's_03_e_20_shot_017358_017432': ['s_03_e_20_shot_017358_017432', 's_03_e_20_shot_017785_017859', 's_03_e_20_shot_018192_018266', 's_03_e_20_shot_018304_018378', 's_03_e_20_shot_018604_018678'], 's_03_e_20_shot_017785_017859': ['s_03_e_20_shot_017785_017859', 's_03_e_20_shot_018192_018266', 's_03_e_20_shot_018304_018378', 's_03_e_20_shot_018604_018678', 's_03_e_20_shot_018754_018828'], 's_03_e_20_shot_018192_018266': ['s_03_e_20_shot_018192_018266', 's_03_e_20_shot_018304_018378', 's_03_e_20_shot_018604_018678', 's_03_e_20_shot_018754_018828', 's_03_e_20_shot_018904_018978'], 's_03_e_20_shot_018304_018378': ['s_03_e_20_shot_018304_018378', 's_03_e_20_shot_018604_018678', 's_03_e_20_shot_018754_018828', 's_03_e_20_shot_018904_018978', 's_03_e_20_shot_019613_019687'], 's_03_e_20_shot_018604_018678': ['s_03_e_20_shot_018604_018678', 's_03_e_20_shot_018754_018828', 's_03_e_20_shot_018904_018978', 's_03_e_20_shot_019613_019687', 's_03_e_20_shot_019833_019907'], 's_03_e_20_shot_018754_018828': ['s_03_e_20_shot_018754_018828', 's_03_e_20_shot_018904_018978', 's_03_e_20_shot_019613_019687', 's_03_e_20_shot_019833_019907', 's_03_e_20_shot_020295_020369'], 's_03_e_20_shot_020931_021005': ['s_03_e_20_shot_020931_021005', 's_03_e_20_shot_021006_021080', 's_03_e_20_shot_021081_021155', 's_03_e_20_shot_021549_021623', 's_03_e_20_shot_021866_021940'], 's_03_e_20_shot_021006_021080': ['s_03_e_20_shot_021006_021080', 's_03_e_20_shot_021081_021155', 's_03_e_20_shot_021549_021623', 's_03_e_20_shot_021866_021940', 's_03_e_20_shot_022121_022195'], 's_03_e_20_shot_021081_021155': ['s_03_e_20_shot_021081_021155', 's_03_e_20_shot_021549_021623', 's_03_e_20_shot_021866_021940', 's_03_e_20_shot_022121_022195', 's_03_e_20_shot_022286_022360'], 's_03_e_20_shot_021549_021623': ['s_03_e_20_shot_021549_021623', 's_03_e_20_shot_021866_021940', 's_03_e_20_shot_022121_022195', 's_03_e_20_shot_022286_022360', 's_03_e_20_shot_022361_022435'], 's_03_e_20_shot_021866_021940': ['s_03_e_20_shot_021866_021940', 's_03_e_20_shot_022121_022195', 's_03_e_20_shot_022286_022360', 's_03_e_20_shot_022361_022435', 's_03_e_20_shot_022814_022888'], 's_03_e_20_shot_022121_022195': ['s_03_e_20_shot_022121_022195', 's_03_e_20_shot_022286_022360', 's_03_e_20_shot_022361_022435', 's_03_e_20_shot_022814_022888', 's_03_e_20_shot_023085_023159'], 's_03_e_20_shot_022286_022360': ['s_03_e_20_shot_022286_022360', 's_03_e_20_shot_022361_022435', 's_03_e_20_shot_022814_022888', 's_03_e_20_shot_023085_023159', 's_03_e_20_shot_023210_023284'], 's_03_e_20_shot_024462_024536': ['s_03_e_20_shot_024462_024536', 's_03_e_20_shot_024552_024626', 's_03_e_20_shot_024695_024769', 's_03_e_20_shot_025388_025462', 's_03_e_20_shot_025688_025762'], 's_03_e_20_shot_024552_024626': ['s_03_e_20_shot_024552_024626', 's_03_e_20_shot_024695_024769', 's_03_e_20_shot_025388_025462', 's_03_e_20_shot_025688_025762', 's_03_e_20_shot_026246_026320'], 's_03_e_20_shot_028787_028861': ['s_03_e_20_shot_028787_028861', 's_03_e_20_shot_028897_028971', 's_03_e_20_shot_029293_029367', 's_03_e_20_shot_029546_029620', 's_03_e_20_shot_029621_029695'], 's_03_e_20_shot_028897_028971': ['s_03_e_20_shot_028897_028971', 's_03_e_20_shot_029293_029367', 's_03_e_20_shot_029546_029620', 's_03_e_20_shot_029621_029695', 's_03_e_20_shot_032395_032469'], 's_03_e_20_shot_029293_029367': ['s_03_e_20_shot_029293_029367', 's_03_e_20_shot_029546_029620', 's_03_e_20_shot_029621_029695', 's_03_e_20_shot_032395_032469', 's_03_e_20_shot_032615_032689'], 's_03_e_20_shot_029546_029620': ['s_03_e_20_shot_029546_029620', 's_03_e_20_shot_029621_029695', 's_03_e_20_shot_032395_032469', 's_03_e_20_shot_032615_032689', 's_03_e_20_shot_033341_033415'], 's_03_e_20_shot_029621_029695': ['s_03_e_20_shot_029621_029695', 's_03_e_20_shot_032395_032469', 's_03_e_20_shot_032615_032689', 's_03_e_20_shot_033341_033415', 's_03_e_20_shot_033561_033635'], 's_03_e_20_shot_032395_032469': ['s_03_e_20_shot_032395_032469', 's_03_e_20_shot_032615_032689', 's_03_e_20_shot_033341_033415', 's_03_e_20_shot_033561_033635', 's_03_e_20_shot_033636_033710'], 's_03_e_20_shot_033561_033635': ['s_03_e_20_shot_033561_033635', 's_03_e_20_shot_033636_033710', 's_03_e_20_shot_033968_034042', 's_03_e_20_shot_034043_034117', 's_03_e_20_shot_034199_034273'], 's_03_e_21_shot_038258_038332': ['s_03_e_21_shot_038258_038332', 's_03_e_21_shot_038795_038869', 's_03_e_21_shot_038995_039069', 's_03_e_21_shot_039193_039267', 's_03_e_21_shot_039939_040013'], 's_03_e_21_shot_038795_038869': ['s_03_e_21_shot_038795_038869', 's_03_e_21_shot_038995_039069', 's_03_e_21_shot_039193_039267', 's_03_e_21_shot_039939_040013', 's_03_e_21_shot_040249_040323'], 's_03_e_21_shot_038995_039069': ['s_03_e_21_shot_038995_039069', 's_03_e_21_shot_039193_039267', 's_03_e_21_shot_039939_040013', 's_03_e_21_shot_040249_040323', 's_03_e_21_shot_040577_040651'], 's_03_e_21_shot_039193_039267': ['s_03_e_21_shot_039193_039267', 's_03_e_21_shot_039939_040013', 's_03_e_21_shot_040249_040323', 's_03_e_21_shot_040577_040651', 's_03_e_21_shot_040652_040726'], 's_03_e_21_shot_039939_040013': ['s_03_e_21_shot_039939_040013', 's_03_e_21_shot_040249_040323', 's_03_e_21_shot_040577_040651', 's_03_e_21_shot_040652_040726', 's_03_e_21_shot_040727_040801'], 's_03_e_21_shot_040249_040323': ['s_03_e_21_shot_040249_040323', 's_03_e_21_shot_040577_040651', 's_03_e_21_shot_040652_040726', 's_03_e_21_shot_040727_040801', 's_03_e_21_shot_041294_041368'], 's_03_e_21_shot_040577_040651': ['s_03_e_21_shot_040577_040651', 's_03_e_21_shot_040652_040726', 's_03_e_21_shot_040727_040801', 's_03_e_21_shot_041294_041368', 's_03_e_21_shot_041613_041687'], 's_03_e_21_shot_040652_040726': ['s_03_e_21_shot_040652_040726', 's_03_e_21_shot_040727_040801', 's_03_e_21_shot_041294_041368', 's_03_e_21_shot_041613_041687', 's_03_e_21_shot_041954_042028'], 's_03_e_21_shot_041613_041687': ['s_03_e_21_shot_041613_041687', 's_03_e_21_shot_041954_042028', 's_03_e_21_shot_042029_042103', 's_03_e_21_shot_042104_042178', 's_03_e_21_shot_042346_042420'], 's_03_e_21_shot_041954_042028': ['s_03_e_21_shot_041954_042028', 's_03_e_21_shot_042029_042103', 's_03_e_21_shot_042104_042178', 's_03_e_21_shot_042346_042420', 's_03_e_21_shot_042863_042937'], 's_03_e_21_shot_042029_042103': ['s_03_e_21_shot_042029_042103', 's_03_e_21_shot_042104_042178', 's_03_e_21_shot_042346_042420', 's_03_e_21_shot_042863_042937', 's_03_e_21_shot_043076_043150'], 's_03_e_21_shot_042104_042178': ['s_03_e_21_shot_042104_042178', 's_03_e_21_shot_042346_042420', 's_03_e_21_shot_042863_042937', 's_03_e_21_shot_043076_043150', 's_03_e_21_shot_043226_043300'], 's_03_e_21_shot_042346_042420': ['s_03_e_21_shot_042346_042420', 's_03_e_21_shot_042863_042937', 's_03_e_21_shot_043076_043150', 's_03_e_21_shot_043226_043300', 's_03_e_21_shot_043525_043599'], 's_03_e_21_shot_042863_042937': ['s_03_e_21_shot_042863_042937', 's_03_e_21_shot_043076_043150', 's_03_e_21_shot_043226_043300', 's_03_e_21_shot_043525_043599', 's_03_e_21_shot_043600_043674'], 's_04_e_25_shot_023155_023229': ['s_04_e_25_shot_023155_023229', 's_04_e_25_shot_023305_023379', 's_04_e_25_shot_023694_023768', 's_04_e_25_shot_023965_024039', 's_04_e_25_shot_024308_024382'], 's_04_e_25_shot_023305_023379': ['s_04_e_25_shot_023305_023379', 's_04_e_25_shot_023694_023768', 's_04_e_25_shot_023965_024039', 's_04_e_25_shot_024308_024382', 's_04_e_25_shot_024607_024681'], 's_04_e_25_shot_023694_023768': ['s_04_e_25_shot_023694_023768', 's_04_e_25_shot_023965_024039', 's_04_e_25_shot_024308_024382', 's_04_e_25_shot_024607_024681', 's_04_e_25_shot_024682_024756'], 's_04_e_25_shot_023965_024039': ['s_04_e_25_shot_023965_024039', 's_04_e_25_shot_024308_024382', 's_04_e_25_shot_024607_024681', 's_04_e_25_shot_024682_024756', 's_04_e_25_shot_024832_024906'], 's_04_e_25_shot_024308_024382': ['s_04_e_25_shot_024308_024382', 's_04_e_25_shot_024607_024681', 's_04_e_25_shot_024682_024756', 's_04_e_25_shot_024832_024906', 's_04_e_25_shot_025076_025150'], 's_04_e_25_shot_024607_024681': ['s_04_e_25_shot_024607_024681', 's_04_e_25_shot_024682_024756', 's_04_e_25_shot_024832_024906', 's_04_e_25_shot_025076_025150', 's_04_e_25_shot_025430_025504'], 's_04_e_25_shot_024682_024756': ['s_04_e_25_shot_024682_024756', 's_04_e_25_shot_024832_024906', 's_04_e_25_shot_025076_025150', 's_04_e_25_shot_025430_025504', 's_04_e_25_shot_025663_025737'], 's_04_e_25_shot_024832_024906': ['s_04_e_25_shot_024832_024906', 's_04_e_25_shot_025076_025150', 's_04_e_25_shot_025430_025504', 's_04_e_25_shot_025663_025737', 's_04_e_25_shot_026169_026243'], 's_04_e_25_shot_025076_025150': ['s_04_e_25_shot_025076_025150', 's_04_e_25_shot_025430_025504', 's_04_e_25_shot_025663_025737', 's_04_e_25_shot_026169_026243', 's_04_e_25_shot_026895_026969'], 's_04_e_25_shot_025430_025504': ['s_04_e_25_shot_025430_025504', 's_04_e_25_shot_025663_025737', 's_04_e_25_shot_026169_026243', 's_04_e_25_shot_026895_026969', 's_04_e_25_shot_027137_027211'], 's_04_e_25_shot_025663_025737': ['s_04_e_25_shot_025663_025737', 's_04_e_25_shot_026169_026243', 's_04_e_25_shot_026895_026969', 's_04_e_25_shot_027137_027211', 's_04_e_25_shot_027212_027286'], 's_04_e_25_shot_026169_026243': ['s_04_e_25_shot_026169_026243', 's_04_e_25_shot_026895_026969', 's_04_e_25_shot_027137_027211', 's_04_e_25_shot_027212_027286', 's_04_e_25_shot_027467_027541'], 's_04_e_25_shot_026895_026969': ['s_04_e_25_shot_026895_026969', 's_04_e_25_shot_027137_027211', 's_04_e_25_shot_027212_027286', 's_04_e_25_shot_027467_027541', 's_04_e_25_shot_027542_027616'], 's_06_e_02_shot_039270_039344': ['s_06_e_02_shot_039270_039344', 's_06_e_02_shot_039358_039432', 's_06_e_02_shot_039589_039663', 's_06_e_02_shot_039739_039813', 's_06_e_02_shot_039875_039949'], 's_06_e_02_shot_039358_039432': ['s_06_e_02_shot_039358_039432', 's_06_e_02_shot_039589_039663', 's_06_e_02_shot_039739_039813', 's_06_e_02_shot_039875_039949', 's_06_e_02_shot_040051_040125'], 's_06_e_02_shot_039589_039663': ['s_06_e_02_shot_039589_039663', 's_06_e_02_shot_039739_039813', 's_06_e_02_shot_039875_039949', 's_06_e_02_shot_040051_040125', 's_06_e_02_shot_040414_040488'], 's_06_e_02_shot_039739_039813': ['s_06_e_02_shot_039739_039813', 's_06_e_02_shot_039875_039949', 's_06_e_02_shot_040051_040125', 's_06_e_02_shot_040414_040488', 's_06_e_02_shot_040564_040638'], 's_06_e_02_shot_039875_039949': ['s_06_e_02_shot_039875_039949', 's_06_e_02_shot_040051_040125', 's_06_e_02_shot_040414_040488', 's_06_e_02_shot_040564_040638', 's_06_e_02_shot_040714_040788']}
    # os.makedirs('unseen_test', exist_ok=True)
    # for story in _tmp:
    #     # make a dir for each story
    #     os.makedirs(os.path.join('unseen_test', story), exist_ok=True)
    #     # clean the dir
    #     for shot in _tmp[story]:
    #         helper.save_images(path=os.path.join('unseen_test', story), id=shot)

