# coding=utf-8
import os

import numpy as np
import torch
from utils.general import LOGGER


##检测框架，集合检测功能
class Detect:
    def __init__(self, opt):
        self.iou = opt.iou_thres
        self.conf = opt.conf_thres

        self.gt_nums = 0
        self.detect_nums = 0
        self.correct_detect_nums = 0

    def compute_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def compare_index(self, xml_info, txt_info):
        for gt_key, gt_value in xml_info.items():
            # 类别以及bbox满足要求：
            # 1. 有gt对应的类别
            # 2. iou满足要求
            for txt_key, txt_value in txt_info.items():
                if txt_key != gt_key:
                    continue

                txt_value = txt_value[txt_value[:, 4] > self.conf]

                iou = self.compute_iou(gt_value, txt_value[:, :4])
                x = torch.where(iou > self.iou)  # filter iou
                # LOGGER.info(f"iou result: {iou}, shape:{iou.shape}")
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
                # LOGGER.info(f"match result:{matches}, nums:{matches.shape[0]}")
                if txt_key == gt_key:  # 正确数量
                    self.correct_detect_nums += n

    def get_index(self, eps=1e-7):

        precision = "{:.4f}".format(self.correct_detect_nums / (self.detect_nums + eps))
        recall = "{:.4f}".format(self.correct_detect_nums / (self.gt_nums + eps))

        data = {"iou": self.iou, "Confidence": self.conf, "correct": self.correct_detect_nums,
                "gt": self.gt_nums, "detect": self.detect_nums, "precision": precision,
                "recall": recall}

        LOGGER.info(
            f'all detect nums: {self.detect_nums},correct detect nums: {self.correct_detect_nums},'
            f' precision:{self.correct_detect_nums / (self.detect_nums + eps) * 100:.2f}%\n'
            f'all gt nums: {self.gt_nums},correct detect nums: {self.correct_detect_nums}, '
            f'recall:{self.correct_detect_nums / (self.gt_nums + eps) * 100:.2f}%\n'
        )

        return data
