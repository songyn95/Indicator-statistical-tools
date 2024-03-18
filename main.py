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
import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Manager

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def create_task(opt, conf_thres=None):
    dataloader, dataset = create_dataloader(opt.Manually_annotate_dir)
    LOGGER.info(f'dataloader sizes {len(dataloader)}')
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
    k = 10
    test_flag = 0

    ##abs path
    opt.source_file = opt.source_file.replace(".\\", os.getcwd() + os.sep, 1) if str(opt.source_file).startswith(
        '.\\') else opt.source_file
    opt.Manually_annotate_dir = opt.Manually_annotate_dir.replace(".\\", os.getcwd() + os.sep, 1) if str(
        opt.Manually_annotate_dir).startswith('.\\') else opt.Manually_annotate_dir

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务到线程池执行
        results = [executor.submit(create_task, opt, i / k * 1.0) for i in range(k)]
        # results = [executor.submit(create_task, 0.0) for i in range(k)]

        # 获取并等待结果
        for f in concurrent.futures.as_completed(results):
            print(f.result())

    # with Manager() as manager:
    #     d = manager.dict()
    #     jobs = [Process(target=create_task, args=(opt, i / k * 1.0)) for i in range(k)]
    #     for j in jobs:
    #         j.start()
    #     for j in jobs:
    #         j.join()

    # with multiprocessing.Pool(processes=k) as pool:
    #     # 使用map函数并行执行任务
    #
    #     results = pool.map(create_task, [(opt, i / k * 1.0) for i in range(k)])
    #
    #     # 输出结果
    #     # for result in results:
    #     print(results)

    plot_evolve(opt.save_csv_path)
    t2 = time.time()
    LOGGER.info(f'cost total time: {(t2 - t1) / 60.0} minutes')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
