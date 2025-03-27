from skrl.utils.logger import Logger
from typing import Any, Dict, List, Optional, Tuple, Union
import collections
import torch
import datetime
import os
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

class CustomLogger():
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__
            )
        self.experiment_dir = os.path.join(directory, experiment_name)
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        self.tracking_data = collections.defaultdict(list)

    def track_data(self, tag: str, value: float) -> None:
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        for k, v in self.tracking_data.items():
            self.writer.add_scalar(k, np.mean(v), timestep)
        self.tracking_data.clear()

    
    