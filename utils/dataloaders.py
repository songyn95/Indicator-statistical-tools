# coding=utf-8
from xml.etree import ElementTree as ET
import glob
from pathlib import Path
import os
import numpy as np

"""数据加载器，加载xml文件"""
from utils.general import LOGGER
from torch.utils.data import dataloader


def create_dataloader(opt, batch=1):
    dataset = Dataset(opt, batch)
    loader = InfiniteDataLoader(dataset, batch)
    return loader, dataset


class Dataset:
    def __init__(self, opt, batch_size=1):
        self.path = opt.Manually_annotate_dir
        self.data_type = opt.data_type
        self.batch_size = batch_size

        ## xml文件
        try:
            f = []  # xml files
            for p in self.path if isinstance(self.path, list) else [self.path]:
                p = Path(p)
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
            raise Exception(f"Error loading data from {self.path}:") from e
        LOGGER.info(f'GT dataloader path: {self.path}\tbatch_size: {self.batch_size} ')

    def __getitem__(self, index):
        id_ = self.im_files[index]
        try:  # utf-8
            anno = ET.parse(id_)
        except:  # gb2312
            eachfile = open(id_, encoding='GB2312')
            anno = ET.parse(eachfile).getroot()

        bbox = list()
        labelname = list()
        frameid = list()
        obj_id = list()

        filename = anno.find('path').text.strip().split('\\')[-1]  ## filename 000001.jpg or 000001.mp4
        if self.data_type == 'video':
            frameid.append(anno.find('framenumber').text)

        for obj in anno.findall('object'):
            if self.data_type == 'video':
                obj_id.append(int(obj.find('id').text))

            bndbox_anno = obj.find('bndbox')
            try:  # images
                bbox.append([int(bndbox_anno.find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            except:  # video to x1，y1,x2，y2
                obj_box = [int(bndbox_anno.find(tag).text) for tag in ('xmin', 'ymin', 'width', 'height')]
                obj_box[-2] += obj_box[0]
                obj_box[-1] += obj_box[1]
                bbox.append(obj_box)
            labelname.append(obj.find('name').text.strip())

        bbox = np.stack(bbox).astype(np.int32)  ##boundingbox
        if self.data_type == 'video':
            frameid = np.stack(frameid).astype(np.uint8)  ##frameid
            obj_id = np.stack(obj_id).astype(np.uint8)  ##obj_id

            return filename, frameid, labelname, obj_id, bbox
        return filename, labelname, bbox

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
