# coding=utf-8
"""Logging utils."""
import os

from utils.general import LOGGER, colorstr

LOGGERS = ("csv", "tb")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = lambda *args: None  # None = SummaryWriter(str)


class Loggers:
    # Loggers class
    def __init__(self, opt=None, logger=None, include=LOGGERS):
        self.save_dir = opt.save_dir
        self.opt = opt
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "IOU",  # 交并比
            "threshold",  # 置信度阈值
            "precision",  # 精度
            "recall",  # 召回率
            "allDetectNum",  # 所有检出数量
            "CorrectNum",  # 正确检出数量
        ]  # params
        self.best_keys = ["precision", "recall", "allDetectNum", "CorrectNum"]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        if "tb" in self.include:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

