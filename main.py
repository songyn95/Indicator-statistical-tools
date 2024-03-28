# coding=utf-8
"""
@project:   blueberry
@File:      main.py
@IDE:
@author:    song yanan
@Date:  2024/3/13 20:04

"""
from pathlib import Path
import sys
import argparse
import os

from utils.general import print_args, LOGGER
from utils.dataloaders import create_dataloader
from utils.handlefile import HandleFile
from tqdm import tqdm
import time
import concurrent.futures

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def create_task(opt, conf_thres=None):
    dataloader, dataset = create_dataloader(opt)
    LOGGER.info(f"dataloader sizes {len(dataloader)}")
    pbar = tqdm(dataloader)
    # 初始化检测器
    file_info = HandleFile(opt)
    if conf_thres is not None:
        file_info.set_conf(conf_thres)
    for i, obj_info in enumerate(pbar):
        file_info.compare(obj_info)
    file_info.write_csv()


def parse_opt():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=str, default=ROOT / "result.txt", help="result file")
    parser.add_argument("--Manually-annotate-dir", type=str, default=ROOT / "gt", help="annotate dir")
    parser.add_argument("--result-file", type=str, default=ROOT / "result" / "result.csv",
                        help="Statistical Results Table, Threshold 0-1")
    parser.add_argument("--data_type", default="image", help="image or video")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=1920, help="img size")
    parser.add_argument("--data-path", type=str, default=ROOT / "test_data" / "image" / "images",
                        help="img or video data path")
    parser.add_argument("--save-data-path", type=str, default=ROOT / "result" / "save_pic",
                        help=" save img or video path")
    parser.add_argument("--save-dir", default=ROOT / "result" / "tensorboard_dirs", help="save tensorboard logs")
    parser.add_argument("--tensorboard", action="store_true", help="view at web ")
    parser.add_argument("--plot", action="store_true", help="plot boundboxs ")
    parser.add_argument("--plot_evolve", action="store_false", help="plot PR, ROC curves ")

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """
    :param opt:
    :return:
    """
    result_file = Path(opt.result_file)
    save_data_path = Path(opt.save_data_path)
    save_dir = Path(opt.save_dir)

    if not result_file.exists():
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.touch()

    if not save_data_path.exists():
        save_data_path.mkdir(parents=True, exist_ok=True)

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    k = 1000

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务到线程池执行
        results = [executor.submit(create_task, opt, i / k * 1.0) for i in range(k)]
        # results = [executor.submit(create_task, opt, 0.0) for i in range(k)]

        # 获取并等待结果
        for f in concurrent.futures.as_completed(results):
            print(f.result())

    if opt.plot_evolve:
        HandleFile.plot_evolve(opt)

    t2 = time.time()
    LOGGER.info(f"cost total time: {(t2 - t1) / 60.0} minutes")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
