from readTrc import Trc
from tqdm import tqdm
from datetime import datetime

import os
import glob
import pandas as pd
import datetime as dt
import numpy as np
import warnings
import json

class TraceReader:

    """
    Class to read data from Teledyne-Lecroy Oscilloscope to pandas dataframe. The default naming convention for saved 
    traces of the Teledyne Lecroy scope follows CX--YYdeg--NNNNN.trc, with X, Y and N denoting the corresponding channel,
    some angle I guess and the index of the recorded waveform. Note that each trace is saved as a single file.
    When calling load_traces for the first time, the traces are aggregated into a single pandas DataFrame and the stored
    under path_to_trc/.. . The presence of a json-file corresponding to the specified name is checked everytime.
    To reload from the individual trc-files delete or move the produced json-file.
    
    Attributes:
        path_to_trc: Path to directory containing PMT output and reference signal traces.
        chn_pmt: Channel number on the oscilloscope corresponding to the input of the PMT
        chn_rf: Channel number on the oscilloscope corresponding to the input of the RF
    """

    def __init__(self, path_to_data: str, range=(None, None)):

        self.import_range = range
        
        self.path_to_data = path_to_data
        
    def load_traces(self, chn: int = -1,  save_json: bool = True):

        """
        Method to load recorded traces from teledyne format to dataframe and save to json.

        Returns:
            df_pmt: Pandas DataFrame containing the timebase, voltage, and trigger timing. columns: 'time', 'voltage', 'trigger_timing'
            df_rf : Pandas DataFrame containing the voltage information of the RF. columns: 'voltage'
        """
      
        if not os.path.isdir(self.path_to_data):
             directory, measurement = os.path.split(self.path_to_data)
             raise Exception('%s not found in %s' %(measurement, directory))

        if chn == -1:
            warnings.warn("chn not passed; Using 1 as default", UserWarning, stacklevel=2)
            chn = 1
        
        # Set paths to search for previously loaded data and for saving first time data
        # 
        #   parent_dir
        #   |--data (path_to_data)
        #   |  |--C*00000.trc
        #   |  |--C*00001.trc
        #   |  | .
        #   |  | .
        #   |  | .
        #   |  |--C*NNNNN.trc
        #   |--data.json

        parent_dir_file_content = next(os.walk(self.path_to_data+'/..'))[2] # dir content of parent dir to list
        parent_dir, trace_dir = os.path.split(self.path_to_data)

        fn = trace_dir+f'_C{chn}.json'
        fn_info = trace_dir+'_DAQInfo.json'

        file_path = os.path.join(parent_dir, fn)
        info_path = os.path.join(parent_dir, fn_info)

        loader = Trc()

        traces_unordered = glob.glob(os.path.join(self.path_to_data, f'C{chn}*.trc'))        
        traces_unordered.sort()
        traces = traces_unordered[self.import_range[0]:self.import_range[1]]
        

        # Check if data was processed previously
        
        if fn in parent_dir_file_content:

            # if corresponding file is found, reads from json.

            print('Fetching Traces from %s' % fn)
            traces = pd.read_json(file_path).iloc[self.import_range[0]: self.import_range[1]]
            df = pd.DataFrame({'trc_index': traces['trc_index'], 'trigger_timing': pd.to_datetime(traces['trigger_timing'], unit="us"), 'time': traces['time'], 'voltage': traces['voltage']})
            
            df['time'] = df.time.apply(lambda v: np.array(v))   
            df['voltage'] = df.voltage.apply(lambda v: np.array(v))
            
            _info = open(info_path) 
            info = json.load(_info)

        else:

            # Else aggregate traces into dataframes:
            print('Collecting CHN%i traces from %s' %(chn, trace_dir))
            # Initialize DataFrames

            df = pd.DataFrame(columns = ['index', 'trc_index', 'time', 'voltage', 'trigger_timing'])

            # Fill it with all the traces

            for i in tqdm(range(len(traces))):  
                
                trc = os.path.split(traces[i])[1][-9:-4] # Fetch trace index pmt

                time, voltage, info = loader.open(traces[i])
                
                timing = info['TRIGGER_TIME']

                df = pd.concat([df, pd.DataFrame(data={'time': [np.array(time)],
                                                        'voltage': [np.array(voltage)],
                                                        'trigger_timing': timing,
                                                        'trc_index': int(trc)})])
                
            df['time'] = df.time.apply(lambda v: np.array(v))   
            df['voltage'] = df.voltage.apply(lambda v: np.array(v))

        # Then save to file

            if save_json:
                df.reset_index(drop=True, inplace=True)

                save_to_file = pd.DataFrame({'time': df['time'],            
                                            'voltage': df['voltage'],
                                            'trc_index': df['trc_index'],
                                            'trigger_timing': df['trigger_timing']})

                save_to_file.to_json(file_path, date_unit='us')

                # Separate file with scope information 

        
                info["TRIGGER_TIME"] = info['TRIGGER_TIME'].isoformat() # Prevents serialization from transforming datetime object to unix timestamp.

                with open(info_path, 'w') as file:
                    json.dump(info,file)      

        return df, info


def readTDC_histogram(self, TDCdir):

    path = os.path.join(self.path_to_data,TDCdir)
    
    df = pd.read_csv(path,
                    sep=";",
                    comment="#",       
                    header=None,
                    names=["bins", "counts"],
                    skipinitialspace=True)
    return df
