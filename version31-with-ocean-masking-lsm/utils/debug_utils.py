import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
from datetime import datetime


class DebugLogger:
    def __init__(self, output_dir="debug_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"debug_{timestamp}.log")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
            handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def log_function_call(self, func_name, args=None, kwargs=None):
        msg = f"CALL {func_name}"
        if args:
            msg += f" args: {args}"
        if kwargs:
            msg += f" kwargs: {kwargs}"
        self.logger.debug(msg)

    def log_data_shape(self, name, data):
        """Log data shape and basic info"""
        if isinstance(data, pd.DataFrame):
            self.logger.debug(
                f"DATA {name}: shape={data.shape}, cols={list(data.columns)}"
            )
            self.logger.debug(f"      non-null: {data.notna().sum().to_dict()}")
        elif isinstance(data, np.ndarray):
            self.logger.debug(f"DATA {name}: shape={data.shape}, dtype={data.dtype}")
        elif isinstance(data, xr.Dataset):
            self.logger.debug(
                f"DATA {name}: vars={list(data.data_vars)}, dims={dict(data.dims)}"
            )

    def log_warning(self, message, data=None):
        """Log warnings with data context"""
        self.logger.warning(message)
        if data is not None:
            self.log_data_shape("Warning Context", data)

    def log_error(self, message, exception=None):
        """Log errors with full traceback"""
        self.logger.error(message)
        if exception:
            self.logger.error(f"Exception: {exception}")
            import traceback

            self.logger.error(traceback.format_exc())
