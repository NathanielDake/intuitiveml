import datetime
import logging
from collections import OrderedDict

import pandas as pd

from notebooks.computer_science.utils.s3.s3_utils import upload_versioned_df_to_s3

STATS_KEY = 'run_stats_{}.csv'

class ETLLogging:
    """
    Helper class to keep track of ETL metrics without introducing too much visual noise.

    To use:
        1) Instantiate logging instance.
        >> logger = ETLLogging()
        2) Decorate function(s) that need to have stats tracked.
        >> @logger.logging_wrapper
        3) Update its return signature to have a final return value of logger.logging_gather_stats(locals())
        >> return ..., ..., logger.logging_gather_stats(locals())
        4) Access the run log dictionary that contains stats, or view it as a dataframe.
        >> logger.run_log_dict
        >> logger.generate_run_log_df()

    Example logging_var_name_map:
      LOGGING_VAR_NAME_MAP = {
          'my_func/my_var': 'Friendly name for my func and var'
      }

    """

    def __init__(self, logging_var_name_map=None, logging_inst=logging):
        self.run_log_dict = {}
        self.logging_var_name_map = OrderedDict(logging_var_name_map)
        self.logging_inst = logging_inst

    def logging_gather_stats(self, local_dict):
        logging_dict = {}
        for key, val in local_dict.items():
            if hasattr(val, '__len__'):
                logging_dict[f'{key}'] = len(val)
        return logging_dict

    def logging_wrapper(self, func):
        def inner(*args, **kwargs):
            func_return = func(*args, **kwargs)
            logging_dict, func_name = func_return[-1], func.__name__
            self.run_log_dict[func.__name__] = logging_dict

            return func_return[:-1][0] if len(func_return[:-1]) == 1 else func_return[:-1]

        return inner

    def generate_run_log_df(self):
        if len(self.run_log_dict) < 1:
            print("Dictionary has no contents.")
            return

        reform = {
            (outerKey, innerKey): values for outerKey, innerDict in self.run_log_dict.items()
            for innerKey, values in innerDict.items()
        }

        df = pd.DataFrame(reform, index=[0]).T.rename_axis(['Function', 'Entity']).rename({0: "Count"},
                                                                                          axis=1).reset_index()

        if (self.logging_var_name_map is None):
            return df

        def determine_sort_order(row):
            for idx, key in enumerate(self.logging_var_name_map):
                if row == key:
                    return int(idx)

        df = df.assign(
            friendly_names=(df.Function + '/' + df.Entity).map(self.logging_var_name_map),
            sort_idx=(df.Function + '/' + df.Entity).apply(determine_sort_order),
            date=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ).sort_values(
            by='sort_idx',
            ascending=True
        ).reset_index(
            drop=True
        )

        return df

    def generate_human_readable_run_log_df(self):
        if self.logging_var_name_map is None:
            return None

        df = self.generate_run_log_df()

        df = df.assign(
            Entity=df.friendly_names
        ).dropna()[['Entity', 'Count']]

        return df

    def send_run_log_dataframe_to_log_file(self):
        """Log the run log dataframe."""

        df_run_log = self.generate_run_log_df()
        self.logging_inst.info('\nRun log Dataframe: \n' + df_run_log.to_string())

        df_run_log_readable = self.generate_human_readable_run_log_df()
        self.logging_inst.info('\nHuman readable run log dataframe: \n' + df_run_log_readable.to_string())

    def send_run_log_dataframe_to_s3(self):

        df = self.generate_run_log_df()
        return upload_versioned_df_to_s3(df, STATS_KEY)
