import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

from matplotlib import cm

class TsmPlotter():

    def __init__(self, df_results):
        """
        TODO Write the docs

        arguments:
            offset: float in (0,1] to shift xaxis, in case distribution of interest sits at the edge of the RF-Period
            df_results: Pandas DataFrame containing timing information computed with TsmAnalysis.ref_timing method.
        """

        self.df = df_results
        if 'ref_timing' not in self.df.columns:
            warnings.warn("The provided data frame does not contain any timing information. Some methods will crash when called." \
            "Run TsmAnalysis(your_df).ref_timing() to ensure proper functionality of all methods in TsmPlotter.")
        pass

    def tsm_hist2d(self, bins=(100, 100), range=None, offset=None, save_fig=False, fig_path=None, fn=None, tick_step = 1e-9):
        """
        TODO Write the docs
        """
        period = 1/np.median(self.df.rf_freq)
        vrange = np.abs(self.df.amp.min() - self.df.amp.max())

        ref_timing = self.df.ref_timing
        
        if range != None:

            dt = range[0][1] - range[0][0]
            dV = range[1][1] - range[1][0]
            
            # Calculate number of bins of full histogram  necessary s.t. the number of specified
            # bins corresponds to the number of bins in the specified range .

            binsx = round(period/dt*bins[0])
            binsy = round(vrange/dV*bins[1])
            print(binsx, binsy)

            xticks = np.arange(range[0][0], range[0][1], tick_step)
            tick_labels = np.rint(xticks/1e-12) # Display xticks in picoseconds
            plt.xticks(xticks, labels=[f"{tick:.0f}" for tick in tick_labels])
            plt.xlim(range[0][0], range[0][1])
            plt.ylim(range[1][0], range[1][1])
        
            plt.xlabel('Time Difference [ps]')
        
        else:
            binsx, binsy = bins
            plt.xlabel('Time Difference [s]')
        
        cmap = plt.cm.viridis
        H, xedges, yedges = np.histogram2d(ref_timing, self.df.amp, bins=[binsx, binsy])
        
        X, Y = np.meshgrid(xedges, yedges)

        if offset != None:
            
            index = int(np.round(np.shape(H)[0]*offset))
            H = np.concat((H[:][index:], H[:][:index]), axis=0)
        
        
       
        cmap.set_bad(color='white')
        
        plt.pcolormesh(X, Y, H.T, norm=cm.colors.LogNorm())
        
       
        plt.colorbar()
        plt.ylabel('Pulse Amplitude [V]')

        if save_fig:
            plt.savefig(os.path.join(fig_path, fn))
        
        plt.show()