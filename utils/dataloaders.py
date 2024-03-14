# coding=utf-8
from xml.etree import ElementTree as ET
import glob
from pathlib import Path
import os
import numpy as np

"""数据加载器，加载xml文件"""
from utils.general import LOGGER
from torch.utils.data import dataloader


def create_dataloader(path, batch=1):
    LOGGER.info(f'GT dataloader path: {path}\tbatch_size: {batch} ')
    dataset = Dataset(path, batch)
    loader = InfiniteDataLoader(dataset, batch)
    return loader, dataset


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
                    LOGGER.info(f'file name:{p}')
                else:
                    raise FileNotFoundError(f"{p} does not exist")

            self.im_files = sorted(f)
            assert self.im_files, f"No xml file found"

        except Exception as e:
            raise Exception(f"Error loading data from {path}:") from e

    def __getitem__(self, index):
        # 逐个获取xml文件
        id_ = self.im_files[index]
        anno = ET.parse(id_)

        bbox = list()
        labelname = list()
        difficult = list()

        file = anno.find('path').text.strip()
        filename = file.split('\\')[-1] ## filename 000001.jpg

        # 获取图像标签大小
        img_sz = [0, 0]  # width height
        for img_info in anno.findall('size'):
            img_sz[0] = int(img_info.find('width').text)
            img_sz[1] = int(img_info.find('height').text)

        for obj in anno.findall('object'):
            if int(obj.find('difficult').text) == 1:  # 0表示易识别，1表示难识别
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # bbox.append([float(bndbox_anno.find(tag).text) / img_sz[1] if tag in ['ymin', 'ymax'] else float(
            #     bndbox_anno.find(tag).text) / img_sz[0] for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            bbox.append([int(bndbox_anno.find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            labelname.append(obj.find('name').text.strip())

        bbox = np.stack(bbox).astype(np.float32)  ##boundingbox
        # label = np.stack(label).astype(np.int32)  #标签读取，需要
        difficult = np.array(difficult, dtype=bool).astype(np.uint8)
        return filename, labelname, bbox, difficult

    def __len__(self):
        return len(self.im_files)


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)
