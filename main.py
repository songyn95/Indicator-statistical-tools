# coding=utf-8
"""
@project:   blueberry
@File:  val.py
@IDE:
@author:    song yanan
@Date:  2024/3/13 20:04

"""
from pathlib import Path
import sys
import argparse
import os

from utils.general import print_args
from utils.loggers import LOGGER
from utils.dataloaders import create_dataloader
from utils.handlefile import HandleFile
from utils.plot import plot_evolve
from tqdm import tqdm
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=str, default=ROOT / "result.txt", help="result file")
    parser.add_argument("--Manually-annotate-dir", type=str, default=ROOT / "gt", help="annotate dir")
    parser.add_argument("--save-csv-path", type=str, default=ROOT / "result" / "result.csv",
                        help="Statistical Results Table, Threshold 0-1")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=1920, help="img size")
    parser.add_argument("--img-path", type=str, default=ROOT / 'test', help="img path")
    parser.add_argument("--save-img-path", type=str, default=ROOT / "result" / 'save_pic', help=" save img path")
    parser.add_argument("--conf-thres-setting", action="store_false", help=" Threshold 0-1 interval 0.001")
    parser.add_argument("--save-dir", default=ROOT / 'exp', help="save to project/name")

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """
    :param opt:
    :return:
    """
    t1 = time.time()
    ##abs path
    opt.source_file = opt.source_file.replace(".\\", os.getcwd() + os.sep, 1) if str(opt.source_file).startswith(
        '.\\') else opt.source_file
    opt.Manually_annotate_dir = opt.Manually_annotate_dir.replace(".\\", os.getcwd() + os.sep, 1) if str(
        opt.Manually_annotate_dir).startswith('.\\') else opt.Manually_annotate_dir

    # evolve_csv = opt.save_csv_path
    # print(evolve_csv)
    # to global path
    if opt.conf_thres_setting:
        for conf_thres in range(0, 1001, 1):
            i_conf_thres = conf_thres / 1000
            opt.conf_thres = i_conf_thres
            # 1. dataloader
            dataloader, dataset = create_dataloader(opt.Manually_annotate_dir)
            LOGGER.info(f'dataloader sizes {len(dataloader)}')
            pbar = tqdm(dataloader)
            # 初始化检测器
            file_info = HandleFile(opt)
            for i, obj_info in enumerate(pbar):
                # xml文件和txt进行对比
                file_info.compare(obj_info)
            file_info.write_csv()

    else:
        # 初始化检测器,只编译一次
        file_info = HandleFile(opt)
        dataloader, dataset = create_dataloader(opt.Manually_annotate_dir)
        LOGGER.info(f'dataloader sizes {len(dataloader)}')
        pbar = tqdm(dataloader)
        for i, obj_info in enumerate(pbar):
            # xml文件和txt进行对比
            file_info.compare(obj_info)
        file_info.write_csv()

    plot_evolve(opt.save_csv_path)

    t2 = time.time()
    LOGGER.info(f'cost total time: {(t2 - t1) / 60.0} mins')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
