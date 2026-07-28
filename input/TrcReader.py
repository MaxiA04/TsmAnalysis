from readTrc import Trc
from tqdm import tqdm


import os
import glob
import pandas as pd
import datetime as dt
import numpy as np

class TraceReader:

    """
    Class to read data from Teledyne-Lecroy Oscilloscope to pandas dataframe. The default naming convention for saved 
    traces of the Teledyne Lecroy scope follows CX--YYdeg--NNNNN.trc, with X, Y and N denoting the corresponding channel,
    some angle I guess and the index of the recorded waveform. Note that each trace is saved as a single file.

    Attributes:
        path_to_trc: Path to directory containing PMT output and reference signal traces.
        chn: Channel on the scope on which the data to be loaded was recorded.
        traces: list containing the sorted filenames (by increasing acquisition index N) of all traces to be loaded into the dataframe
    """

    def __init__(self, path_to_data: str, chn: int, verbose=True):
        
        self.path_to_data = path_to_data
        self.chn = str(chn)
        traces_unordered = glob.glob(os.path.join(path_to_data, 'C'+self.chn+'*.trc'))        
        traces_unordered.sort()
        self.traces = traces_unordered

    def load_traces(self, fetch_timing_info=False):

        df = pd.DataFrame(columns = ['index', 'time', 'voltage', 'chn', 'trigger_timing'])

        loader = Trc()
        
        if fetch_timing_info:

            for i in tqdm(range(len(self.traces))):  

                time, voltage, info = loader.open(self.traces[i])
                
                timing = info['TRIGGER_TIME']

                df = pd.concat([df, pd.DataFrame(data={'index': i, 'time': [list(time)],
                                                        'voltage': [list(voltage)],'chn':  int(self.chn), 'trigger_timing': timing})])

        else:

            for i in tqdm(range(len(self.traces))):            
                time, voltage, info = loader.open(self.traces[i])

                timing = np.nan

                df = pd.concat([df, pd.DataFrame(data={'index': i, 'time': [list(time)],
                                                'voltage': [list(voltage)],'chn':  int(self.chn), 'trigger_timing': timing})])
        
        df.set_index('index', inplace=True)

        return df, info