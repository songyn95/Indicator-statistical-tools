# coding=utf-8
import functools
import os

import pandas as pd

from utils.general import LOGGER
from collections import defaultdict
from Algorithm_indicators.detection.detect import Detect
from Algorithm_indicators.recognition.recongnize import Recongnize
from Algorithm_indicators.classification.classify import Classify
import numpy as np
import torch
from utils.plot import write_to_csv, plot_labels

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory


class HandleFile:
    img_list = []

    def __init__(self, opt):
        self.img_path = opt.img_path
        self.save_img_path = opt.save_img_path
        self.save_csv_path = opt.save_csv_path

        self.file = opt.source_file
        self.fileinfo = self.readfile()

        self.detect = Detect(opt)
        self.recongnize = Recongnize(opt)
        self.classify = Classify(opt)

    def set_conf(self, conf_thres=None):
        if conf_thres is not None:
            self.detect.conf = conf_thres

    def readfile(self):
        data = []
        try:
            with open(self.file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(line.rstrip().split(' '))

            data = pd.DataFrame(data)

        except Exception as e:
            LOGGER.error(f"open file failed, error message:{e}")
            raise FileExistsError(f"{self.file} open failed!!")

        return data

    def compare(self, obj_info):
        filename = obj_info[0][0]  # 000001 filename
        obj_info[1] = [i[0] for i in obj_info[1]]  # class

        class_txt = dict()
        class_xml = defaultdict(list)

        matched_rows = self.fileinfo[self.fileinfo[0].str.contains(filename)].index.tolist()
        LOGGER.info(f"match:{matched_rows}{type(matched_rows)}")
        if not matched_rows:
            LOGGER.error(f"error No XML found for txt:{filename}")
            raise FileNotFoundError(f"{filename} does not exist")

        rows = matched_rows[0]

        # 检测总数量
        fileinfo = self.fileinfo[rows:rows + 1].dropna(axis=1)
        size = fileinfo.size
        assert not (size - 1) % 6, LOGGER.error(f"failed file data:{fileinfo}")

        class_txt.clear()
        class_xml.clear()

        data_len = (size - 1) // 6
        self.detect.detect_nums += data_len
        class_info = fileinfo.iloc[:, 1:]
        # 处理line： filename class bbox conf:
        # 00001.jpg, car, 0.1,0.1, 0.1, 0.1, 0.6 car, 0.1,0.1, 0.1, 0.1, 0.6
        for j in range(0, size - 1, 6):
            key = class_info.iloc[0, j]
            img_info = class_info.iloc[:, j + 1:j + 6].values.astype('float').flatten()
            if key and key not in class_txt:
                class_txt[key] = [img_info]
            else:
                class_txt[key] += [img_info]

        # tensor
        for txt_key in class_txt:
            class_txt[txt_key] = torch.Tensor(np.stack(class_txt[txt_key]).astype(np.float32))

        # xml handle: filename, labelname, bbox, difficult:
        # 00001, [car,dog], [[0.1,0.1, 0.1, 0.1],[0.1,0.1, 0.1, 0.1]], [0,0]
        obj_info[2] = obj_info[2].squeeze().tolist()  # 降维
        for key, value in zip(obj_info[1], obj_info[2]):
            class_xml[key].append(value)

        class_xml = dict(class_xml)  # {"car":[[],[]]}
        for xml_key, xml_value in class_xml.items():
            # gt数量
            self.detect.gt_nums += len(xml_value)
            class_xml[xml_key] = torch.Tensor(np.stack(xml_value).astype(np.float32))

        LOGGER.info(f"txt info:{class_txt}")
        LOGGER.info(f"xml info:{class_xml}")

        # 写信息到图像
        if filename not in HandleFile.img_list:
            file = os.path.join(self.img_path, filename)
            plot_labels(class_xml, class_txt, file, self.save_img_path)
            HandleFile.img_list.append(filename)

        # IOU
        self.detect.compare_index(class_xml, class_txt)

    def write_csv(self):
        data = self.detect.get_index()
        write_to_csv(data, self.save_csv_path)
