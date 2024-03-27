# coding=utf-8
"""
@project:   blueberry
@File:      detect.py
@IDE:
@author:    song yanan
@Date:  2024/3/15 23:34

"""
import os

import numpy as np
import torch
from utils.general import LOGGER
from utils.general import colorstr


##检测框架，集合检测功能
class Detect:
    def __init__(self, opt):
        self.iou = opt.iou_thres
        self.conf = opt.conf_thres
        self.data_type = opt.data_type  # images or video
        # 图片检测
        self.gt_nums = 0
        self.detect_nums = 0  # 总检测数
        self.correct_detect_nums = 0  # 正确检测数
        self.fp = 0  # 误报
        # 抓拍
        self.detect_capture_nums = 0
        self.correct_capture_nums = 0
        self.capture_lists = []
        self.gt_lists = []
        self.correct_capture_lists = []

    def set_conf(self, conf_thres):
        self.conf = conf_thres

    def compute_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def compare_index(self, xml_info, txt_info):
        for key, value in txt_info.items():
            if self.data_type == "video":
                self.capture_lists.append(key)  # ('person',0) 总抓拍列表
                self.detect_capture_nums += value.reshape(-1, 5).shape[0]  # 总抓拍数
            else:
                self.detect_nums += value.reshape(-1, 5).shape[0]

        for gt_key, gt_value in xml_info.items():
            gt_shape = gt_value.reshape(-1, 4).shape
            if self.data_type == "video":  # video每帧内id不重复,只需要添加gt_key
                self.gt_lists.append(gt_key)
            else:
                # 总gt数量
                self.gt_nums += gt_shape[0]

            for txt_key, txt_value in txt_info.items():
                txt_shape = txt_value.reshape(-1, 5).shape
                txt_value = txt_value[txt_value[:, 4] > self.conf]
                iou = self.compute_iou(gt_value, txt_value[:, :4])
                x = torch.where(iou > self.iou)  # filter iou

                # LOGGER.info(f"iou result: {iou}, shape:{iou.shape}")
                # LOGGER.info(f"x result: {x}")
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()

                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                else:
                    matches = np.zeros((0, 3))
                n = matches.shape[0]
                if gt_key == txt_key:
                    if self.data_type == "image":
                        self.correct_detect_nums += n
                        self.fp += txt_shape[0] - n  # 误报
                    else:
                        self.correct_capture_nums += n
                        self.correct_capture_lists.append(gt_key)  # 一次只比对一个id（n==1），故直接添加即可

    def get_index(self, eps=1e-7):
        if self.data_type == "image":
            precision = round(self.correct_detect_nums / (self.detect_nums + eps), 4)
            recall = round(self.correct_detect_nums / (self.gt_nums + eps), 4)
            fpr = round(self.fp / (self.detect_nums + eps), 4)
            fnr = round((1 - recall), 4)

            data = {"iou": self.iou, "Confidence": self.conf, "correct": self.correct_detect_nums,
                    "gt": self.gt_nums, "detect": self.detect_nums, "FPR": fpr, "FNR": fnr, "precision": precision,
                    "recall": recall}

            LOGGER.info(
                f'all gt nums: {colorstr(self.gt_nums)}, all detect nums: {colorstr(self.detect_nums)}, '
                f'correct nums: {self.correct_detect_nums}, false positive nums: {colorstr(self.fp)}, '
                'false positive rate:' + colorstr(f"{fpr:.2%}") + ', false negative rate:' + colorstr(
                    f"{fnr:.2%}") + '\nprecision: ' + colorstr(f"{precision:.2%}") + ', recall: ' + colorstr(
                    f"{recall:.2%}") + '\n'
            )
        else:
            # 1. 抓拍召回率:（正确抓拍真值数-重复抓拍真值数）/ 总真值数
            # 2. 抓拍检准率: 正确抓拍真值数 / 总抓拍目标数
            # 3. 抓拍重复率: 重复抓拍真值数 / 正确抓拍真值数

            capture_recall = round(len(set(self.correct_capture_lists)) / len(set(self.gt_lists)), 4)
            capture_precision = round(self.correct_capture_nums / self.detect_capture_nums, 4)
            capture_repetition = round(
                (self.correct_capture_nums - len(set(self.correct_capture_lists))) / self.correct_capture_nums, 4)

            data = {"iou": self.iou, "Confidence": self.conf, "detect": self.detect_capture_nums,
                    "gt": len(set(self.gt_lists)), "correct": self.correct_capture_nums,
                    "capture_repetition": capture_repetition, "capture_precision": capture_precision,
                    "capture_recall": capture_recall}

            LOGGER.info(
                f'all gt capture nums:{colorstr(len(set(self.gt_lists)))}, all capture nums:'
                f'{colorstr(self.detect_capture_nums)}, correct capture nums:{colorstr(self.correct_capture_nums)}, '
                f'repetition capture nums:{colorstr(self.correct_capture_nums - len(set(self.correct_capture_lists)))}\n'
                'capture repetition:' + colorstr(f"{capture_repetition:.2%}") + ", capture precision: " + colorstr(
                    f"{capture_precision:.2%}") + ', capture recall: ' + colorstr(
                    f"{capture_recall:.2%}") + '\n'
            )

        return data
