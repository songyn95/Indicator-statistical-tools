#coding=utf-8
"""Logging utils."""
import os

import pkg_resources as pkg

from utils.general import LOGGER, colorstr
from utils.loggers.clearml.clearml_utils import ClearmlLogger
from utils.loggers.wandb.wandb_utils import WandbLogger
# from utils.plots import plot_images, plot_labels, plot_results

LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = lambda *args: None  # None = SummaryWriter(str)

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version("0.12.2") and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

try:
    import clearml

    assert hasattr(clearml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None

try:
    if RANK in {0, -1}:
        import comet_ml

        assert hasattr(comet_ml, "__version__")  # verify package import not local dir
        from utils.loggers.comet import CometLogger
    else:
        comet_ml = None
except (ImportError, AssertionError):
    comet_ml = None

#csv统计结果表
class Loggers:
    # Loggers class
    def __init__(self, save_dir=None, opt=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.opt = opt
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "IOU", #交并比
            "threshold", #置信度阈值
            "precision",#精度
            "recall",   #召回率
            "allDetectNum",#所有检出数量
            "CorrectNum",  #正确检出数量
        ]  # params
        self.best_keys = ["precision", "recall", "allDetectNum", "CorrectNum"]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Messages
        if not comet_ml:
            prefix = colorstr("Comet: ")
            s = f"run 'pip install comet_ml' to automatically track and visualize runs in Comet"
            self.logger.info(s)

        # TensorBoard
        s = self.save_dir
        if "tb" in self.include:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

            # W&B
            if wandb and "wandb" in self.include:
                self.opt.hyp = self.hyp  # add hyperparameters
                self.wandb = WandbLogger(self.opt)
            else:
                self.wandb = None

            # ClearML
            if clearml and "clearml" in self.include:
                try:
                    self.clearml = ClearmlLogger(self.opt, self.hyp)
                except Exception:
                    self.clearml = None
                    prefix = colorstr("ClearML: ")
                    LOGGER.warning(
                        f"{prefix}WARNING ⚠️ ClearML is installed but not configured, skipping ClearML logging."
                        f" See https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration#readme"
                    )

            else:
                self.clearml = None

            # Comet
            if comet_ml and "comet" in self.include:
                if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                    run_id = self.opt.resume.split("/")[-1]
                    self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)

                else:
                    self.comet_logger = CometLogger(self.opt, self.hyp)

            else:
                self.comet_logger = None
    @property
    def remote_dataset(self):
        """Fetches dataset dictionary from remote logging services like ClearML, Weights & Biases, or Comet ML."""
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict

        return data_dict

def web_project_name(project):
    # Convert local project name to web project name
    if not project.startswith("runs/train"):
        return project
    suffix = "-Classify" if project.endswith("-cls") else "-Segment" if project.endswith("-seg") else ""
    return
