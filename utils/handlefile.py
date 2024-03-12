# coding=utf-8
import os
from utils.general import LOGGER
from PIL import Image
from collections import defaultdict
from Algorithm_indicators.detection.detect import Detect
import numpy as np
import torch
from utils.plot import Annotator,Colors
import cv2
class HandleFile:
    def __init__(self, opt):
        self.iou = opt.iou_thres
        self.conf = opt.conf_thres
        self.img_path = opt.img_path
        self.save_img_path = opt.save_img_path

        self.file = opt.source_file
        self.fileinfo = self.readfile()
        self.detect = Detect(opt)

    def readfile(self):
        try:
            with open(self.file, 'r') as f:
                fileinfo = f.readlines()

        except Exception as e:
            LOGGER.error(f"open file failed, error message:{e}")

        finally:
            return fileinfo

    def write_img(self):
        pass

    def compare(self, obj_info):
        filename = obj_info[0][0]  # 000001 filename
        obj_info[1] = [i[0] for i in obj_info[1]]  # class

        class_txt = dict()
        class_xml = defaultdict(list)

        # 3. exist  find match txt xml
        for i, line in enumerate(self.fileinfo):  # result.txt
            LOGGER.info(f"Handle line number:{i}, need match filename:{filename}")
            LOGGER.info(f"line:{line}")

            if (filename.split('.')[0] in line):

                class_txt.clear()
                class_xml.clear()

                line_list = line.split()[1:]
                assert not len(line_list)% 6, LOGGER.error(f"failed file data:{line}")

                # 处理line： filename class bbox conf:
                # 00001.jpg, car, 0.1,0.1, 0.1, 0.1, 0.6 car, 0.1,0.1, 0.1, 0.1, 0.6
                for j in range(0, len(line_list), 6):
                    img_info = [float(ii) for ii in line_list[j+1:j+6]]
                    if line_list[j] not in class_txt:
                        class_txt[line_list[j]] = [img_info]
                    else:
                        class_txt[line_list[j]] += [img_info]

                #tensor
                for txt_key in class_txt:
                    class_txt[txt_key] = torch.Tensor(np.stack(class_txt[txt_key]).astype(np.float32))

                # xml: filename, labelname, bbox, difficult:
                # 00001, [car,dog], [[0.1,0.1, 0.1, 0.1],[0.1,0.1, 0.1, 0.1]], [0,0]
                for key, value in zip(obj_info[1], obj_info[2]):
                    class_xml[key].append(value)

                class_xml = dict(class_xml)#{"car":[[],[]]}


                LOGGER.info(f"txt info:{class_txt}")
                LOGGER.info(f"xml info:{class_xml}")

                # 写信息到图像
                self.plot_labels(class_xml, class_txt, filename)
                break
        else:
            LOGGER.error(f"error No XML found for txt:{filename}{self.file}")

        # IOU
        self.detect.compare_index(class_xml, class_txt)

    def get_index(self):
        self.detect.get_index()

    def plot_labels(self, xml_info, txt_info, filename):
        """Plots dataset labels, saving correlogram and label images, handles classes, and visualizes bounding boxes."""
        colors = Colors()
        img = os.path.join(self.img_path, filename)
        img = cv2.imread(img)
        annotator = Annotator(img)
        class_dict = {'car':'0', 'sedan':'1'}

        img_sz = [2304,1296]
        for gt_key, gt_value in xml_info.items():
            # plot
            gt_value = gt_value[0]
            gt_value[:, 0] = gt_value[:, 0] * img_sz[0]
            gt_value[:, 1] = gt_value[:, 1] * img_sz[1]
            gt_value[:, 2] = gt_value[:, 2] * img_sz[0]
            gt_value[:, 3] = gt_value[:, 3] * img_sz[1]
            cls = class_dict[gt_key]
            color = colors(cls)
            print(f"gt_value1111:{gt_value}")
            for j, box in enumerate(gt_value.tolist()):
            # gt_key = f"{cls}" if gt_key else f"{cls} {conf[j]:.1f}"
                annotator.box_label(box, cls, color=(255,0,0))

        for txt_key, txt_value in txt_info.items():
            # plot
            print(f"txt_info......")
            cls = class_dict[txt_key]

            txt_value[:4, 0] = txt_value[:4, 0] * img_sz[1]
            txt_value[:4, 1] = txt_value[:4, 1] * img_sz[0]
            txt_value[:4, 2] = txt_value[:4, 2] * img_sz[1]
            txt_value[:4, 3] = txt_value[:4, 3] * img_sz[0]
            print(f"txt_value:{txt_value}{txt_value.shape}")
            col1, col2,col3, col4 = txt_value[:, 0], txt_value[:, 1],txt_value[:, 2],txt_value[:, 3]
            txt_value = torch.tensor(np.vstack((col2,col1,col4,col3,txt_value[:, 4]))).T



            print(f"txt_value:{txt_value}")
            print(f"box[j][4]:{txt_value.tolist()}")

            for j, box in enumerate(txt_value.tolist()):
                print(box)
                print(f"{cls} {str(box[4])}")
                cls = f"{cls} {str(box[4])}"
                print(cls)
                annotator.box_label(box, cls, color=(0,255,0))

        cv2.imwrite(os.path.join(self.save_img_path, filename), annotator.im)  # save



















