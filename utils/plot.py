# coding=utf-8
"""
@project:   blueberry
@File:      plot.py
@IDE:
@author:    song yanan
@Date:  2024/3/14 22:48

"""

from PIL import Image, ImageDraw
from utils.general import is_ascii
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
import cv2
import csv
import pandas as pd
from pathlib import Path
import torch


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        # c = self.palette[int(i) % self.n]
        c = self.palette[len(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        # assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        # self.pil = pil or not is_ascii(example) or is_chinese(example)
        self.pil = pil
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            # self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
            #                        size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        # self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        self.lw = 1  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to images with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to images (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to images (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated images as array
        return np.asarray(self.im)


def write_to_csv(data, save_csv_path):
    """Writes json data for an images to a CSV file, appending if the file exists."""
    with open(save_csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not save_csv_path.is_file():
            writer.writeheader()
        writer.writerow(data)


def plot_evolve(evolve_csv="result.csv"):
    """
    Plots hyperparameter evolution results from a given CSV, saving the plot and displaying best results.

    Example: from utils.plots import *; plot_evolve()
    """
    evolve_csv = Path(evolve_csv)
    df_data = pd.read_csv(evolve_csv, header=None)
    data = df_data.sort_values(by=1)
    x = data.values[:, ]
    fig = plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc("font", **{"size": 8})

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x[:, -2], x[:, -1])
    ax1.set_title('PR curves')

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(x[:, -3][::-1], x[:, -1][::-1])
    ax1.set_title('ROC curves')
    # plt.plot(x[:, -2], x[:, -1])
    # plt.title(f"iou:(0-1) PR curves")
    f = evolve_csv.with_suffix(".png")  # filename
    fig.savefig(f, dpi=400)
    print(f"Saved {f}")


def plot_labels(xml_info, txt_info, filename, save_img_path, data_type="images", tb=None):
    """Plots dataset labels, saving correlogram and label images, handles classes, and visualizes bounding boxes."""
    # colors = Colors()
    img = filename
    img = cv2.imread(img)
    annotator = Annotator(img)
    # images width, height
    height, width = img.shape[:2]
    img_sz = [width, height]
    # add true bbox

    for gt_key, gt_value in xml_info.items():
        # (xmin, ymin, xmax, ymax)
        for j, box in enumerate(gt_value.tolist()):
            annotator.box_label(box, str(gt_key), color=(255, 0, 0))

    # add pred bbox
    for txt_key, txt_value in txt_info.items():
        # plot# plot（xmin, ymin, xmax, ymax）
        if torch.all(txt_value[:4] < 1):  # 反归一化
            txt_value[:, 0] = txt_value[:, 0] * img_sz[0]
            txt_value[:, 1] = txt_value[:, 1] * img_sz[1]
            txt_value[:, 2] = txt_value[:, 2] * img_sz[0]
            txt_value[:, 3] = txt_value[:, 3] * img_sz[1]

        for j, box in enumerate(txt_value.tolist()):
            annotator.box_label(box, str(txt_key), color=(0, 0, 255))

    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    file = os.path.join(save_img_path, filename.split(os.sep)[-1])
    cv2.imwrite(file, annotator.im)  # save

    if tb:
        tb.add_image(Path(file).stem, cv2.imread(str(file))[..., ::-1], dataformats="HWC")
