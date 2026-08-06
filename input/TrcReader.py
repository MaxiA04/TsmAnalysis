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

    def __init__(self, path_to_data: str, chn_pmt: int, chn_rf: int):
        
        self.path_to_data = path_to_data
        self.chn_pmt, self.chn_rf = str(chn_pmt), str(chn_rf)
        
        traces_unordered_pmt = glob.glob(os.path.join(path_to_data, 'C'+self.chn_pmt+'*.trc'))        
        traces_unordered_pmt.sort()
        self.traces_pmt = traces_unordered_pmt

        traces_unordered_rf = glob.glob(os.path.join(path_to_data, 'C'+self.chn_rf+'*.trc'))        
        traces_unordered_rf.sort()
        self.traces_rf = traces_unordered_rf

    def load_traces(self):

        """
        Method to load the recorded traces of the PMT and RF signal simultaneously.

        Returns:
            df_pmt: Pandas DataFrame containing the timebase, voltage, and trigger timing. columns: 'time', 'voltage', 'trigger_timing'
            df_rf : Pandas DataFrame containing the voltage information of the RF. columns: 'voltage'
        """

        
        # Set paths to search for previously loaded data and for saving first time data

        parent_dir_file_content = next(os.walk(self.path_to_data+'/..'))[2]
        parent_dir, trace_dir = os.path.split(self.path_to_data)

        fn = trace_dir+'.json'
        fn_info = trace_dir+'_DAQInfo.json'

        file_path = os.path.join(parent_dir, fn)
        info_path = os.path.join(parent_dir, fn_info)

        loader = Trc()

        # Check if data was processed previously
        
        if fn in parent_dir_file_content:

            # if corresponding file is found, reads from json.

            print('Fetching Traces from %s' % os.path.join(parent_dir,fn))
            traces = pd.read_json(file_path)
            df_pmt = pd.DataFrame({'time': traces['time'], 'trigger_timing': pd.to_datetime(traces['trigger_timing'], unit="us"), 'voltage': traces['pmt']})
            df_rf = pd.DataFrame({'voltage': traces['rf']})

            _info = open(info_path) 
            info = json.load(_info)

        else:

            # Else aggregate traces into dataframes:
            print('Collecting traces from %s' % self.path_to_data )
            # Initialize DataFrames

            df_pmt = pd.DataFrame(columns = ['index', 'time', 'voltage', 'trigger_timing'])
            df_rf = pd.DataFrame(columns = ['index', 'voltage'])

            # Fill it with all the traces

            for i in tqdm(range(len(self.traces_pmt))):  
                
                pmt_trc = os.path.split(self.traces_pmt[i])[1][-9:-4] # Fetch trace index pmt

                time, voltage, info = loader.open(self.traces_pmt[i])
                
                timing = info['TRIGGER_TIME']

                df_pmt = pd.concat([df_pmt, pd.DataFrame(data={'time': [list(time)],
                                                        'voltage': [list(voltage)], 'trigger_timing': timing})])

                rf_trc = os.path.split(self.traces_rf[i])[1][-9:-4] # Fetch trace index rf
                _time, voltage, _info = loader.open(self.traces_rf[i])
                df_rf = pd.concat([df_rf, pd.DataFrame(data={'voltage': [list(voltage)]})])

                # And check that both traces have matching indices

                if pmt_trc != rf_trc:
                    warnings.warn(f'Trace index in PMT channel {pmt_trc} does not match RF channel {rf_trc}')

        # Then save to file
        
            df_pmt.reset_index(drop=True, inplace=True)
            df_rf.reset_index(drop=True, inplace=True)

            save_to_file = pd.DataFrame({'time': df_pmt['time'],
            
                                        'pmt': df_pmt['voltage'],
                                        'trigger_timing': df_pmt['trigger_timing'],
                                        'rf': df_rf['voltage']})

            save_to_file.to_json(file_path, date_unit='us')

            # Separate file with scope information 

            info["TRIGGER_TIME"] = info['TRIGGER_TIME'].isoformat() # Prevents serialization from transforming datetime object to unix timestamp.

            with open(info_path, 'w') as file:
                        json.dump(info,file)

        # Replace Index columns
            
        
        

        return df_pmt, df_rf, info

