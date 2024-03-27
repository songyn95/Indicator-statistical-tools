# coding=utf-8
"""
@project:   blueberry
@File:  json_txt.py
@IDE:   
@author:    song yanan 
@Date:  2024/3/25 16:28

"""
import os.path

'''
将json告警信息转为txt
'''
import json
import argparse
import glob
from pathlib import Path


def parse_opt():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/txt", help="log dirs")
    # parser.add_argument("--result-dir", type=str, default="logs/result", help="result dir")

    opt = parser.parse_args()
    return opt


def convert_json_txt(txt, temp_file):
    "json_path: 需要处理的json文件的路径"
    "txt_path: 将json文件处理后存放所需的txt文件名"
    with open(txt, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            write_bbox(data, temp_file)


def write_bbox(data, temp_file):
    line = str(data.get('frameId')) + '.jpg '
    # 提取字段
    for obj in data["detections"]:
        track_id = str(obj["trackId"])  # 将带有跟踪的id映射
        x_min, y_min, x_max, y_max = obj["box"]["x0"], obj["box"]["y0"], obj["box"]["x1"], obj["box"]["y1"]
        detectScore = obj.get('detectScore')
        line += ' '.join([track_id, str(x_min), str(y_min), str(x_max), str(y_max), str(detectScore)])
        line += ' '

    temp_file.write(line + '\n')


def main(opt):
    f = []  # xml files
    log_dir = Path(opt.log_dir)
    if log_dir.is_dir():  # dir
        f += glob.glob(str(log_dir / "*.txt"))

    for each in f:
        if not os.path.exists('result'):
            os.mkdir('result')
        temp_file = open(str(Path('result') / each.split('\\')[-1]), 'a', encoding='utf-8')
        convert_json_txt(each, temp_file)
        temp_file.close()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
