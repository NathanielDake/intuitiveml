import sys; sys.path.insert(0, '/app/work/') 
import datetime
import logging
from collections import defaultdict, OrderedDict

import pandas as pd

from notebooks.computer_science.utils.s3.s3_utils import (
    upload_versioned_df_to_s3,
    upload_versioned_pickle_to_s3
)

STATS_KEY = 'run_stats_{}.csv'


class ETLLogging(object):
    """
    Function to log information related to the local variables of given function. 
    
    Usage:
    - Decorate an entry point function (ex. main)
    - Any functions whose locals you are interested in, pass their names at 
      instantiation via internal_funcs_to_log_list
    - Note: The wrapped function is included in this list by default. If that
      is not desired, on instantiation pass log_wrapped_func=False
    """
    
    default_entity_type_aggregation_map = {
        'DataFrame': 'size',
        'Series': 'size',
        'ndarray': 'size',
        'list': 'size',
        'dict': 'size',
        'str': 'identity',
        'int': 'identity',
        'float': 'identity', 
    }

    default_aggregation_processor_map = {
        'size': lambda x: len(x),
        'identity': lambda x: x 
    }
    
    def __init__(
        self, 
        internal_funcs_to_log_list=[], 
        log_wrapped_func=True, 
        logging_var_name_map=None,
        logging_inst=logging,
        entity_type_aggregation_map=default_entity_type_aggregation_map, 
        aggregation_processor_map=default_aggregation_processor_map,
        
    ):
        self.internal_funcs_to_log_list = internal_funcs_to_log_list
        self.log_wrapped_func = log_wrapped_func
        self.logging_var_name_map = OrderedDict(logging_var_name_map)
        self.logging_inst = logging_inst
        self.entity_type_aggregation_map = entity_type_aggregation_map
        self.aggregation_processor_map = aggregation_processor_map
        self.run_log_dict = {}
            
    def logging_wrapper(self, func):
        
        if self.log_wrapped_func:
            self.internal_funcs_to_log_list += [func.__name__]
        
        def inner(*args, **kwargs):
            
            def tracer(frame, event, arg):
                if (event=='return') and (frame.f_code.co_name in self.internal_funcs_to_log_list):
                    self.run_log_dict[frame.f_code.co_name] = frame.f_locals.copy()
                    
            sys.setprofile(tracer)
            try:
                res = func(*args, **kwargs)
            finally:
                sys.setprofile(None)
            return res

        return inner
    
    def dump_run_log_dict_to_s3(self):
        upload_versioned_pickle_to_s3(self.run_log_dict, STATS_KEY)
        
    def generate_run_log_df(self):
    
        mutable_run_log_dict = defaultdict(dict)
        
        for func_key in self.run_log_dict.keys():
            for var_key, var_val in self.run_log_dict[func_key].items():

                agg_map_key = self.entity_type_aggregation_map[type(var_val).__name__]
                mapped_var_val = self.aggregation_processor_map[agg_map_key](var_val)

                mutable_run_log_dict[func_key].update({
                    var_key: {'var_aggregation_map': agg_map_key, 'mapped_var_value': mapped_var_val}
                })
                
        reform = {
            (outerKey, innerKey): values for outerKey, innerDict in mutable_run_log_dict.items() 
            for innerKey, values in innerDict.items()
        }

        df = pd.DataFrame(reform).T.rename_axis(['function_name', 'variable_name']).reset_index()
        
        if (self.logging_var_name_map is None):
            return df
        
        def determine_sort_order(row):
            for idx, key in enumerate(self.logging_var_name_map):
                if row == key:
                    return int(idx)

        df = df.assign(
            friendly_names=(df.function_name + '/' + df.variable_name).map(self.logging_var_name_map),
            sort_idx=(df.function_name + '/' + df.variable_name).apply(determine_sort_order),
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
        ).rename(columns={'mapped_var_value': 'Value'}).dropna()

        return df[['Entity', 'Value']]
    
    def send_run_log_dataframe_to_log_file(self):
        """Log the run log dataframe."""

        pd.options.display.max_colwidth = 150
        
        df_run_log = self.generate_run_log_df()
        self.logging_inst.info('\nRun log Dataframe: \n' + df_run_log.to_string())
        
        df_run_log_readable = self.generate_human_readable_run_log_df()
        self.logging_inst.info('\nHuman readable run log dataframe: \n' + df_run_log_readable.to_string())
        
    def send_run_log_dataframe_to_s3(self):
        
        df = self.generate_run_log_df()
        return upload_versioned_df_to_s3(df, STATS_KEY)
        