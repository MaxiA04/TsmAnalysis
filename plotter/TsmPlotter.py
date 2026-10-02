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

            df_results: Pandas DataFrame containing timing information computed with TsmAnalysis.ref_timing method.
        """

        self.df = df_results
        if 'ref_timing' not in self.df.columns:
            warnings.warn("The provided data frame does not contain any timing information. Some methods will crash when called." \
            "Run TsmAnalysis(your_df).ref_timing() to ensure proper functionality of all methods in TsmPlotter.")
        pass

    def tsm_hist2d(self, bins=(100, 100), colx='ref_timing', coly='amp', range=None, offset=None, save_fig=False, fig_path=None, fn=None, tick_step = 1e-9):

        """
        TODO Write the docs

        offset: float in (0,1) to shift xaxis, in case distribution of interest sits at the edge of the RF-Period
        """
        period = 1/np.median(self.df.rf_freq)
        vrange = np.abs(self.df.amp.min() - self.df.amp.max())

        xdata = self.df[colx]
        ydata = self.df[coly]

        mask = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[mask]
        ydata = ydata[mask]
        
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
        
            plt.xlabel(colx)
        
        else:
            binsx, binsy = bins
            plt.xlabel(colx)
        
        cmap = plt.cm.viridis
        H, xedges, yedges = np.histogram2d(xdata, ydata, bins=[binsx, binsy])
        
        X, Y = np.meshgrid(xedges, yedges)

        if offset != None:
            
            index = int(np.round(np.shape(H)[0]*offset))
            H = np.concat((H[:][index:], H[:][:index]), axis=0)
        
        
       
        cmap.set_bad(color='white')
        
        plt.pcolormesh(X, Y, H.T, norm=cm.colors.LogNorm())
        
       
        plt.colorbar()
        plt.ylabel(coly)

        if save_fig:
            plt.savefig(os.path.join(fig_path, fn))
        
        plt.show()

    def plot_hist(self, col, nbins, range):

        hist, bins = np.histogram(self.df[col], bins=nbins, range=range)
        bin_centers = (bins[:-1] + bins[1:])/2

        plt.errorbar(bin_centers, hist, yerr=np.sqrt(hist), color='k', fmt='.')
        plt.xlabel(col)
        plt.ylabel('Counts')
        plt.show()

    def plot_fit_performance(self, cols: list=['rf_amp', 'rf_freq', 'rf_phase'],
                             
                            nbins: list=[60, 60, 20],
                            ranges: None=[(1.26, 1.3), (50.05e6, 51e6), (-np.pi, np.pi)],
                            nalign: int= 4,
                            ):
        """ 
        Plots collection of essential hists of extracted fit parameters.

        Args:
            cols: List of columns to produce histograms for.
            nbins: List of number of bins corresponding in same order as cols
            ranges: List of touples of ranges in same order as cols
        """
        self.plot_cfd_corrected_pulses()

        for i in range(len(cols)):
            self.plot_hist(col=cols[i], nbins=nbins[i], range=(ranges[i][0], ranges[i][1]))

        
        for i in self.df.index[:nalign]:

            self.plot_rf_fits(i)

    def plot_cfd_corrected_pulses(self, ntraces=50):

        fig, ax = plt.subplots()

        ax.set_prop_cycle(color=plt.cm.Oranges(np.linspace(0, 1, 16)))

        for i in range(ntraces):
            if i == 10:
                ax.plot(self.df.time[i], self.df.pmt[i], alpha=0.4, label='Original LED trigger')
            ax.plot(self.df.time[i], self.df.pmt[i], alpha=0.4)

        ax.set_prop_cycle(color=plt.cm.Blues(np.linspace(0, 1, 16)))

        for i in range(50):
            if i == 10:
                ax.plot(self.df.CFD_corrected_time[i], self.df.pmt[i], alpha=.4, label='CFD corrected')
            ax.plot(self.df.CFD_corrected_time[i], self.df.pmt[i], alpha=.4)

        plt.title('Spread in pulse arrival times - LED v. CFD')
        plt.ylabel('Voltage [V]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.show()

    def plot_rf_fits(self, i: int, ylim = (-1.3, 1.3)):

        rf_model = lambda t, A, B, w, phi: A*np.sin(w*t + phi) + B
        
        tb = np.arange(-2e-8, max(self.df.CFD_corrected_time[i]), 5e-11)
                    
        popt = self.df.rf_amp[i], self.df.rf_bsl_shift[i], 2*np.pi*self.df.rf_freq[i], self.df.rf_phase[i]
        plt.scatter(self.df.CFD_corrected_time[i], self.df.ref[i], marker='.', color='orange', alpha=1, label='RF Data')
        plt.plot(tb, rf_model(tb, *popt),  label='model')
        plt.plot(self.df.CFD_corrected_time[i], self.df.pmt[i], color='C3', label='PMT pulse')

        plt.fill_betweenx(ylim, -self.df.ref_timing[i], 0, color='gray',alpha=0.2)
        plt.axvline(-self.df.ref_timing[i], color='r', linestyle=':', label='Ref start')
        plt.axvline(0, color='k', linestyle='--', label='Pulse_start')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.ylim(ylim)
        plt.legend(loc='upper left')
        plt.show()

