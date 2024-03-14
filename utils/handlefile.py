# coding=utf-8
import os
from utils.general import LOGGER
from collections import defaultdict
from Algorithm_indicators.detection.detect import Detect
from Algorithm_indicators.recognition.recongnize import Recongnize
from Algorithm_indicators.classification.classify import Classify
import numpy as np
import torch
from utils.plot import Annotator, Colors
import cv2
import csv
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory


class HandleFile:
    def __init__(self, opt):
        self.img_path = opt.img_path
        self.save_img_path = opt.save_img_path
        self.save_csv_path = opt.save_csv_path

        self.file = opt.source_file
        self.fileinfo = self.readfile()

        self.detect = Detect(opt)
        self.recongnize = Recongnize(opt)
        self.classify = Classify(opt)

    def readfile(self):
        try:
            with open(self.file, 'r') as f:
                fileinfo = f.readlines()

        except Exception as e:
            LOGGER.error(f"open file failed, error message:{e}")

        finally:
            return fileinfo

    def compare(self, obj_info):
        filename = obj_info[0][0]  # 000001 filename
        obj_info[1] = [i[0] for i in obj_info[1]]  # class

        class_txt = dict()
        class_xml = defaultdict(list)

        # 3. exist find match txt xml
        for i, line in enumerate(self.fileinfo):  # result.txt
            if filename.split('.')[0] in line:

                class_txt.clear()
                class_xml.clear()

                line_list = line.split()[1:]
                assert not len(line_list) % 6, LOGGER.error(f"failed file data:{line}")
                # 检测总数量
                self.detect.detect_nums += len(line_list) // 6

                # 处理line： filename class bbox conf:
                # 00001.jpg, car, 0.1,0.1, 0.1, 0.1, 0.6 car, 0.1,0.1, 0.1, 0.1, 0.6
                for j in range(0, len(line_list), 6):
                    img_info = [float(ii) for ii in line_list[j + 1:j + 6]]
                    if line_list[j] not in class_txt:
                        class_txt[line_list[j]] = [img_info]
                    else:
                        class_txt[line_list[j]] += [img_info]

                # tensor
                for txt_key in class_txt:
                    class_txt[txt_key] = torch.Tensor(np.stack(class_txt[txt_key]).astype(np.float32))

                # xml: filename, labelname, bbox, difficult:
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
                self.plot_labels(class_xml, class_txt, filename)
                break
        else:
            LOGGER.error(f"error No XML found for txt:{filename}{self.file}")
            raise FileNotFoundError(f"{filename} or {self.file} does not exist")

        # IOU
        self.detect.compare_index(class_xml, class_txt)

    def get_index(self):
        self.detect.get_index()

    def write_to_csv(self):
        """Writes json data for an image to a CSV file, appending if the file exists."""
        data = self.detect.get_index()
        with open(self.save_csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not self.save_csv_path.is_file():
                writer.writeheader()
            writer.writerow(data)

    def plot_evolve(self):
        """
        Plots hyperparameter evolution results from a given CSV, saving the plot and displaying best results.

        Example: from utils.plots import *; plot_evolve()
        """
        evolve_csv = Path(self.save_csv_path)
        data = pd.read_csv(evolve_csv)
        keys = [x.strip() for x in data.columns]
        x = data.values[:, ]
        plt.figure(figsize=(10, 12), tight_layout=True)
        matplotlib.rc("font", **{"size": 8})

        plt.plot(x[:, -2], x[:, -1])
        plt.title(f"iou:(0-1) PR curves")
        f = evolve_csv.with_suffix(".png")  # filename
        plt.savefig(f, dpi=200)
        plt.close()
        print(f"Saved {f}")

    def plot_labels(self, xml_info, txt_info, filename):
        """Plots dataset labels, saving correlogram and label images, handles classes, and visualizes bounding boxes."""
        colors = Colors()
        img = os.path.join(self.img_path, filename)
        img = cv2.imread(img)
        annotator = Annotator(img)
        # image width, height
        height, width = img.shape[:2]
        img_sz = [width, height]

        # add true bbox
        for gt_key, gt_value in xml_info.items():
            # plot（xmin, ymin, xmax, ymax）
            for j, box in enumerate(gt_value.tolist()):
                annotator.box_label(box, gt_key, color=(255, 255, 0))

        # add pred bbox
        for txt_key, txt_value in txt_info.items():
            # plot# plot（xmin, ymin, xmax, ymax）
            if torch.all(txt_value[:4] < 1):  # 反归一化
                txt_value[:, 0] = txt_value[:, 0] * img_sz[0]
                txt_value[:, 1] = txt_value[:, 1] * img_sz[1]
                txt_value[:, 2] = txt_value[:, 2] * img_sz[0]
                txt_value[:, 3] = txt_value[:, 3] * img_sz[1]

            for j, box in enumerate(txt_value.tolist()):
                cls = f"{txt_key} {str(box[4])}"
                annotator.box_label(box, txt_key, color=(255, 255, 0))

        cv2.imwrite(os.path.join(self.save_img_path, filename), annotator.im)  # save

    def get_result(self):
        self.write_to_csv()
        self.plot_evolve()
