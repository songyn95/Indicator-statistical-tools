# coding=utf-8
"""
@project:   blueberry
@File:      handlefile.py
@IDE:
@author:    song yanan
@Date:  2024/3/13 22:59

"""
import functools
import os

import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import cv2

from utils.general import LOGGER
from Algorithm_indicators.detection.detect import Detect
from Algorithm_indicators.recognition.recongnize import Recongnize
from Algorithm_indicators.classification.classify import Classify
from utils.plot import write_to_csv, plot_labels
from utils.general import colorstr
from utils.plot import plot_evolve

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory


class HandleFile:
    img_list = []

    def __init__(self, opt):
        self.data_path = opt.data_path
        self.save_path = opt.save_data_path
        self.result_file = opt.result_file  # save csv path
        self.data_type = opt.data_type  # images or video
        # TensorBoard
        self.save_dir = opt.save_dir
        self.plot = opt.plot
        self.tb = None
        if opt.tensorboard:
            prefix = colorstr("TensorBoard: ")
            LOGGER.info(
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/")

            self.tb = SummaryWriter(str(self.save_dir))

        self.file = opt.source_file
        self.fileinfo, self.file_lines = self.readfile()

        self.detect = Detect(opt)
        self.recongnize = Recongnize(opt)
        self.classify = Classify(opt)

    def set_conf(self, conf_thres=None):
        if conf_thres is not None:
            self.detect.set_conf(conf_thres)

    def readfile(self):
        data = []
        try:
            with open(self.file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(line.rstrip().split(' '))
            data = pd.DataFrame(data)

        except Exception as e:
            LOGGER.error(f"open file failed, error message:{e}")
            raise FileExistsError(f"{self.file} open failed!!")

        return data, len(lines)

    def compare(self, obj_info):
        # video： filename, frameid, labelname, obj_id, bbox
        # images： filename, labelname, bbox
        k = 2 if self.data_type == "video" else 1  # 前 1，2 个为数据名称信息

        # 000001 filename
        filename = obj_info[0][0]
        # class
        obj_info[k] = [j[0] for j in obj_info[k]]
        frameid = obj_info[1].flatten()[0].item() if self.data_type == "video" else None
        # 初始化类别信息
        class_txt = dict()
        class_xml = defaultdict(list)
        # 匹配行
        if self.data_type == "image":
            matched_rows = self.fileinfo[self.fileinfo[0].str.contains(filename)].index.tolist()
        else:
            matched_rows = self.fileinfo[self.fileinfo.iloc[:, 1] == str(frameid)].index.tolist()

        # LOGGER.info(f"match:{matched_rows}{type(matched_rows)}")
        if len(matched_rows) > 1:
            LOGGER.warning(f"match len more than 1:{len(matched_rows)}")
        if not matched_rows:
            LOGGER.error(f"error No XML found for txt:{filename},len:{len(matched_rows)}")
            raise FileNotFoundError(f"{filename} does not exist")

        # 取行
        rows = matched_rows[0]
        fileinfo = self.fileinfo[rows:rows + 1].dropna(axis=1)
        size = fileinfo.size

        data_len = size - 2 if self.data_type == "video" else size - 1
        interval = 7 if self.data_type == "video" else 6
        # LOGGER.info(f"data_len:{data_len}, interval: {interval}")
        assert not data_len % interval, LOGGER.error(f"failed file data:{fileinfo}")

        # 过滤出的行
        class_info = fileinfo.iloc[:, k:]

        if self.data_type == "image":
            # txt
            for j in range(0, data_len, interval):
                key = (class_info.iloc[0, j],)  # (class, id)
                img_info = class_info.iloc[:, j + 1:j + interval].values.astype('float').flatten()

                if key and key not in class_txt:
                    class_txt[key] = [img_info]
                else:
                    class_txt[key] += [img_info]
            # xml
            obj_info[k + 1] = obj_info[k + 1].reshape(-1, 4).tolist()  # 降维

            for key, value in zip(obj_info[k], obj_info[k + 1]):
                class_xml[(key,)].append(value)

        else:  # video
            # txt
            for j in range(0, data_len, interval):
                key = (class_info.iloc[0, j], int(class_info.iloc[0, j + 1]))  # (class, id)
                img_info = class_info.iloc[:, j + 2:j + interval].values.astype('float').flatten()
                if key and key not in class_txt:
                    class_txt[key] = [img_info]
                else:
                    class_txt[key] += [img_info]
            # xml
            obj_info[k + 1] = obj_info[k + 1].reshape(-1, 4).tolist()  # id降维
            obj_info[k + 2] = obj_info[k + 2].reshape(-1, 4).tolist()  # bbox降维

            for key, value1, value2 in zip(obj_info[k], obj_info[k + 1], obj_info[k + 2]):
                class_xml[(key, value1)].append(value2)

        class_xml = dict(class_xml)
        for xml_key, xml_value in class_xml.items():
            class_xml[xml_key] = torch.Tensor(np.stack(xml_value).astype(np.float32))

            # tensor
        for txt_key, txt_value in class_txt.items():
            class_txt[txt_key] = torch.Tensor(np.stack(txt_value).astype(np.float32))

        # LOGGER.info(f"txt info:{class_txt}")
        # LOGGER.info(f"xml info:{class_xml}")
        if self.plot:
            if self.data_type == "image":  # 图像名命名
                if filename not in HandleFile.img_list:
                    HandleFile.img_list.append(filename)
                    file = Path(self.data_path) / filename
                    plot_labels(class_xml, class_txt, str(file), self.save_path, self.data_type, self.tb)

                    if self.file_lines == len(HandleFile.img_list) and self.tb:
                        self.tb.close()
            else:  # video 以帧号命名的图像
                if frameid not in HandleFile.img_list:
                    HandleFile.img_list.append(frameid)
                    file = Path(self.data_path) / (str(frameid) + '.jpg')
                    plot_labels(class_xml, class_txt, str(file), self.save_path, self.data_type, self.tb)

                    if self.file_lines == len(HandleFile.img_list) and self.tb:
                        self.tb.close()

        # IOU
        self.detect.compare_index(class_xml, class_txt)

    def write_csv(self):
        data = self.detect.get_index()
        write_to_csv(data, self.result_file)

    @staticmethod
    def plot_evolve(opt):
        evolve_csv = opt.result_file
        plot_evolve(evolve_csv)
        # add image
        file = evolve_csv.with_suffix(".png")
        if opt.tensorboard:
            tb = SummaryWriter(str(opt.save_dir))
            tb.add_image('0_' + Path(file).stem, cv2.imread(str(file))[..., ::-1], dataformats="HWC")
            tb.close()
