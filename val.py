"""
@project:   blueberry
@File:  val.py
@IDE:   
@author:    song yanan 
@Date:  2024/3/13 12:54

"""
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from xml.etree import ElementTree as ET
import glob
import numpy as np
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory


def plot_evolve(evolve_csv=ROOT / "result.csv"):
    evolve_csv = Path(evolve_csv)
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


class Dataset:
    def __init__(self, path, batch_size=1):
        self.path = path

        ## xml文件
        try:
            f = []  # xml files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.xml"), recursive=True)
                elif p.is_file() and p.name.endswith('.xml'):  # file
                    f += p  #
                else:
                    raise FileNotFoundError(f"{p} does not exist")

            self.im_files = sorted(f)
            assert self.im_files, f"No xml file found"

        except Exception as e:
            raise Exception(f"Error loading data from {path}:") from e

    def convert_txt(self):
        # 逐个获取xml文件
        for eachfile in self.im_files:
            anno = ET.parse(eachfile)
            filename = anno.find('path').text.strip().split(os.sep)[-1]  ## filename 000001.jpg
            # 获取图像标签大小
            img_sz = [0, 0]  # width height
            for img_info in anno.findall('size'):
                img_sz[0] = int(img_info.find('width').text)
                img_sz[1] = int(img_info.find('height').text)
            print(filename)
            filename += ' '
            for obj in anno.findall('object'):
                if int(obj.find('difficult').text) == 1:  # 0表示易识别，1表示难识别
                    continue

                difficult = int(obj.find('difficult').text)
                bndbox_anno = obj.find('bndbox')
                bbox = ' '.join([bndbox_anno.find(tag).text for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
                conf = round(float(bbox.split()[-1]) / img_sz[1], 2)
                labelname = obj.find('name').text.strip()
                filename += ' '.join([labelname, bbox, str(conf)])
                filename += ' '
            # print(str_line)
            f.write(filename + '\n')

    def __len__(self):
        return len(self.im_files)


if __name__ == '__main__':
    # plot_evolve()
    dataset = Dataset(ROOT / "test" / "gt")
    with open(ROOT / "test" / 'pred.txt', 'w') as f:
        dataset.convert_txt()
