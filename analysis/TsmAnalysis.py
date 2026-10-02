import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import chi2
import warnings



    
class TsmAnalysis:

    
    """
    Processing data acquired with the time structure monitors at HIPA.
    TODO Write the docs!

    Attributes:
        compute:            Performs basic computations on input data adding fields [amp, area, min_loc, pulse_start] computing pmt pulse amplitude, 
                            area, peak position and trigger position.
        rf_model:           Model sine-wave for fitting the reference signal.
        falling_edge_model: Linear model used to shape the pmts falling edge
        phase_to_time:      Method to compute timing information from phase estimate of rf-fit.
        optimizer:          wrapper for scipy.optimize function with default boundaries.
        chi_square_test:    Goodnes-of-fit-test for the estimate rf fit parameters. (Should be very well constrained)
        get_rf_timing:      Method to estimate pulse timing with respect to rf-signal. Calls rf_model(), phase_to_time(), optimizer(), chi_square_test()
        CFD_correction:     Method to compute discriminator-walk-corrected pulse start. Calls falling_edge_model().
    """

    def __init__(self, df_pmt_out: pd.DataFrame, df_rf_out: pd.DataFrame, bsl_window=(0, 100)):
        
        self.df_pmt_out = df_pmt_out
        self.df_rf_out = df_rf_out
        self.df = self.compute(df_pmt_out, df_rf_out, bsl_window)
        self.bsl_window = bsl_window

    @staticmethod
    def compute(df_sig_in, df_ref_in, bsl_window):
        """
        Performs basic data manipulations on scope data of PMT output and reference signal.
        Args:
            df_out: Empty Pandas DataFrame to be filled with 
            df_sig_in: Pandas DataFrame containing the traces of the PMT output pulses in the column 'voltage'.
            df_ref_in: Pandas DataFrame containing the traces of the Accelerator cavity reference signal in the column 'voltage'.
        """

        # Merge sig_in and ref_in according to trigger_timing to ensure proper matching of events
        # First make sure there are no invalid entries in trigger_timing columns then merge

        df_sig_in['trigger_timing'] = pd.to_datetime(df_sig_in.trigger_timing, errors='coerce')
        df_ref_in['trigger_timing'] = pd.to_datetime(df_ref_in.trigger_timing, errors='coerce')
        merged = pd.merge(df_sig_in, df_ref_in, on='trigger_timing', how='inner', suffixes=('_pmt', '_rf'))

        df = pd.DataFrame()
        # Trace and trigger timing information
        df['time'] = merged['time']
        df['trigger_timing'] = merged['trigger_timing']

        # PMT trace & calculations 
        df['pmt'] = merged['voltage_pmt']
        df['amp'] = df['pmt'].apply(lambda v: np.min(v))
        df['area'] = df['pmt'].apply(lambda v: np.sum(-np.array(v)[bsl_window[1]:]) )
        df['min_loc'] = df['pmt'].apply(lambda v: np.where(v == np.min(v))[0][0])

        pulse_start = []
        for i in tqdm(df.index):
            time_range = np.abs(df.time[i])
            time_zero = np.min(time_range)
            pulse_start.append(time_zero)

        df['pulse_start'] = pulse_start

        # RF Trace
        df['ref'] = merged['voltage_rf']

        # Deprecated calculations
        # df['voltage'] = df_sig_in['voltage']
        # df['bsl'] = df_sig_in['voltage'].apply(lambda v: np.median(v[bsl_window[0]:bsl_window[1]]))
        # df['rms'] = df_sig_in['voltage'].apply(lambda v: np.sqrt(np.mean(np.square(v[:len(v)//3]))))
        # df['pmt'] = df.apply(lambda v: list(np.array(v.voltage) - v.bsl), axis=1)

        return df
    
    def rf_model(self, t, A, B, w, phi):
        """
        Sinewave reflecting RF reference signal from accelerator cavities.
        Amplitude and frequency are fixed in principle, but allowed to float to 
        absorb fluctuations.
        """
        return A*np.sin(w*t + phi) + B
    
    def falling_edge_model(self, t, m, q):
        """
        Model for falling of PMT pulses. Should have been lambda x, m, q: m*x+q, but here we are. 
        """
        return m*t+q
    

    def phase_to_time(self, phase, freq):
        """
        Method to convert the estimated phase of the RF-signal w.r.t. to the PMT pulse to a time difference. As the phase is 
        a value between -pi and pi, and an intuitive understanding of the pulse arrival w.r.t. the RF-cavitiy signal suggests the
        0-phase instances of the RF be placed at the origin and the pulse at some positive time. To achieve this,
        time estimates greater than 0 are first subtracted by a full period.

        Arguments:
            phase: Phase estimated of a given trace.
            freq: Frequency estimate of a given trace.
        returns:
            time: Seperation in time between the pulse and 0-phase of RF signal, with pulse at origin.
        """
        
        period = 1/freq
      
        time = -phase*period/(2*np.pi)

        if time > 0:
            time -= period

        return time

    def slope_sign(self, x, y, index:int, sep:int = 10):
        """
        Simple method to determine the local behaviour of data surrounding a provided index, i.e. is the first derrivative at 
        index positive or negative over a provided range sep.
        
        Arguments:
            x: Array representing the x-coordinate of the data
            y: Array representing the y-coordinate of the data
            index: Location to be probed
            sep: Distance in index used to calculate linear approx. of slope
        Returns:
            sign: -1, 1 according to the sign of the slope.
        """
        sign = 0

        try:
            slope = (y[index + sep] - y[index])/(x[index + sep] - x[index])
        except:
            slope = (y[index] - y[index - sep])/(x[index] - x[index - sep])

        if slope < 0:
            sign = -1
        elif slope >= 0:
            sign = 1
        elif slope == 0:
            warnings.warn("Slope unsigned. Something's fishy")

        return sign

    def phase_estimator(self, time, rf_trace, frequency=50632229.26542789):
        """
        Finds an initial estimate for the phase parameter of the sinusoidal RF signal, by finding the signal's last zero-crossing before
        the trigger, and checking the sign of the slope at this location.
        Arguments:
                    time: Timebase used to compute separation with corresponding trigger located at the origin
                    rf_trace: Single RF trace, for which the time separation is to be computed
                    freq: Frequencty estimate of the RF.
        returns:
                    phase: Phase estimate in (-pi, pi) range
        """
        # Constrain values to 12 ns intervall centered at origin. In this way the maximum number of zero crossings is 2

        mask = (time > -6e-9) & (time < 6e-9)

        time_arr, ref_arr = time[mask], rf_trace[mask]

        index_zero_xing = np.where(ref_arr**2 == np.min(ref_arr**2))[0][0] # Choose a zero crossing at random.

        sign = self.slope_sign(time_arr, ref_arr, index=index_zero_xing)

        rf_time_estimate = time_arr[index_zero_xing]

        phase0 = -2*np.pi*rf_time_estimate*frequency

        if sign < 0 and phase0 < 0:
            phase0 += np.pi
        elif sign < 0 and phase0 > 0:
            phase0 -= np.pi
            
        return phase0
        
    def optimizer(self, model, time_arr, sig_arr, init_params=None,
                   bounds=None):
        """
        Wrapper for curve_fit plus GoF test
        """
        try:
            popt, pcov = curve_fit(model, time_arr, sig_arr, p0=init_params, bounds=bounds, absolute_sigma=True)
            chi2, dof = self.chi_square_test(time_arr, sig_arr, popt)
            reduced_chi2 = chi2/dof

        except:
            popt, pcov = np.nan, np.nan
            reduced_chi2 = np.nan

        return popt, pcov, reduced_chi2
    
    def get_rf_timing(self, timing='corrected', cfd_bounds=(.1, .9)):
        """
        Method to compute the time difference between the 0-phase of the sinusoidal reference signal and the pulse arrival time of the PMT.
        The latter corresponds to the 0-coordinate of the scope's time base, as this is the instance the trigger is fired. The timing is calculated
        from fitting the reference signal with the PMT signals time base and extracting the phase (∈ (-pi, pi]), which is subsequently translated to 
        a positive definite time separation dt ∈ (0 , 1/f], where f corresponds to the estimated RF frequency, by calling the phase_to_time method.
        The scope's trigger follows the typical leading edge scheme, firing only when a fixed threshold is crossed, causing discriminator walk.       
        To correct for discriminator walk, select the option timing="corrected". In this case the method is CFD_correction is called.
        For scope-inherent LED trigger choose "original"
        edit cfd_bounds (default (0.1, 0.9), corresponding to the definition of pulse rise time in Hamamatsu's PMT Handbook 4.E)

        Arguments:
            timing: string identifier accepts 'corrected' and 'original'. Select between scope-native timebase, and rise-time,i.e. CFD corrected, timebase.
            cfd_bounds: touple with values in the range (0,1). Default: (.1,.9). The first entry must be smaller than the last and ideally symmetric. Defines the voltage boundaries used
            in the CFD correction. Only used if timing = 'corrected'.
        """
       
        if timing == 'corrected':
                   
            if cfd_bounds[0] > cfd_bounds[1] or cfd_bounds[0] < 0 or cfd_bounds[1] > 1:
                warnings.warn('provided cfd_bounds (%f, %f) are invalid. Make sure both are in range (0,1) and the first boundary is lower than the second. The default values will be used instead' %cfd_bounds)
                cfd_bounds = (.1, .9)
            self.CFD_correction(lower=cfd_bounds[0], upper=cfd_bounds[1])            
            timing_col = 'CFD_corrected_time'

        elif timing == 'original':
            timing_col = 'time'
        elif timing not in ['corrected', 'original']:
            warnings.warn(f'timing identifier %s not recognized. The computation defaults to scope-native timebase and is prone to discriminator walk.' %timing)
            timing_col = 'time'

        else:
            print('using scope native pulse timing. (Leading edge trigger)')

        print('Calculating pulse timing w.r.t. RF')

        ref_t = []
        ref_init_t = []

        rchi=[]
        rf_amp = []
        rf_freq = []
        rf_phase = []
        rf_init_phase = []
        rf_bsl_shift = []

        fit_fail_counter = 0

        
        for i in tqdm(self.df.index):

            init_amp = 0.5*(np.max(self.df.ref[i] + np.abs(np.min(self.df.ref[i]))))

            

            time_arr = np.array(self.df[timing_col][i])
            ref_arr = np.array(self.df.ref[i])

            init_phase = self.phase_estimator(time_arr, ref_arr)

            #  [A,B, w, phi]
            init_params = [init_amp, 0, 2*np.pi*50.6e6, init_phase]
            
            try:              
                
                bounds = ([1.26, -1e-2, 3.1e8, -np.inf], [1.3, 1e-2, 3.1825e8, np.inf])

            
                popt, pcov, reduced_chi2 = self.optimizer(self.rf_model, time_arr, ref_arr,
                                                 init_params=init_params, bounds=bounds)

                amp, bsl_shift, ang_freq, phase = popt
                freq = ang_freq/2/np.pi
               
            except:
                fit_fail_counter += 1
                amp, bsl_shift, freq, phase, reduced_chi2= np.nan, np.nan, np.nan, np.nan, np.nan
    
            dt = self.phase_to_time(phase, freq)
            init_dt = self.phase_to_time(init_phase, freq=50632229.26542789)

            ref_t.append(dt)
            ref_init_t.append(init_dt)

            rchi.append(reduced_chi2)
            
            rf_amp.append(amp)
            rf_freq.append(freq)
            rf_phase.append(phase)
            rf_init_phase.append(init_phase)
            rf_bsl_shift.append(bsl_shift)

        self.df['ref_timing'] = -1*np.array(ref_t)
        self.df['ref_init_timing'] = -1*np.array(ref_init_t)

        self.df['rf_amp'] = rf_amp
        self.df['rf_freq'] = rf_freq
        self.df['rf_phase'] = rf_phase
        self.df['rf_init_phase'] = rf_init_phase
        self.df['rf_bsl_shift'] = rf_bsl_shift

        self.df['reduced_chi2'] = rchi
        self.df['fit_misalignment'] = self.df.apply(lambda v: self.phase_to_time(v.reduced_chi2, v.rf_freq), axis=1)

        print('No estimate preduced in %i instances' % fit_fail_counter)
        
    def chi_square_test(self, x, y, popt):
        """
        Method to evaluate goodness of fit. As the error on the voltage reading are not included at the moment (i.e. $\sigma_i$ = 1),
        the typical interpretation of quality fits exhibiting reduced chi-squared values close to unity is not yet valid. As of right now, 
        the reduced chi_square estimated from chi_square/dof somehow corresponds to the phase-shift of the model with respect to the actual data.

        Arguments:
            x: Numpy Array - Shared time base of the model and data
            y: Numpy Array - Measured values
            popt: best fit parameters estimated in optimizer method
        Returns:
            chi_square: Numpy Float64 - Chi square of the model estimated with sigma_i = 1
            dof: degrees of freedom associated to fit
            p: p-value estimate corresponding to fit.
        """

        vertical_resolution = 8/2**12 # 8V dynamic range and 12-bit resolution
        chi_square = np.sum(y - self.rf_model(x, *popt))/vertical_resolution**2
        
        dof = len(y) - len(popt)

        return chi_square, dof

    def CFD_correction(self, lower=.1, upper=.9):
        '''
        Method to correct for amplitude dependent arrival time of pulses caused by leading edge triggering of scope.
        The falling edge fo the pulse is modelled by a linear function. the intersect of the function with the bsl-level of each trace
        is set to be the new corrected 0-coordinate of the time base. The voltage range, over which the falling edge is modelled, is defined by
        the arguments lower and upper.
        Adds a column to the Pandas DataFrame with the updated timing information and rise time.
        
        Arguments:
           lower: Numpy float64, defines the lower bound of the falling edge to be modelled. Must be lower than upper.
           upper: Numpy float64, defines the upper bound of the falling edge to be modelled.

        '''
        corrected_time = []
        delay = []
        risetime = []

        print('Estimating corrected pulse arival times')
        fail_counts = 0
        for i in tqdm(self.df.index):

            peak_position = self.df.min_loc[i]

            amp_10p = lower*self.df.amp[i]
            amp_90p = upper*self.df.amp[i]

            full_amp = np.array(self.df.pmt[i][:peak_position])
            full_time = np.array(self.df.time[i][:peak_position])

            mask = (full_amp < amp_10p) & (full_amp > amp_90p) 

            amp = full_amp[mask]
            time = full_time[mask]

            try:
                p0 = [(amp.min() - amp.max())/(time.max() - time.min()), -.5]
                bounds = ([-np.inf, -np.inf],[0, np.inf])

                popt, pcov, = curve_fit(self.falling_edge_model, time, amp, p0=p0, bounds=bounds) # TODO Add field to df with estimate uncertainty from fit for resolution estimate
                pulse_delay_index = np.where(np.abs(self.falling_edge_model(full_time,*popt)) == np.min(np.abs(self.falling_edge_model(full_time,*popt))))[0][0]
                pulse_delay = full_time[pulse_delay_index]
                
                correct = self.df.time[i] - pulse_delay

            except:
                popt = (np.nan, np.nan)
                pulse_delay_index = np.nan
                pulse_delay = np.nan
                fail_counts += 1

                correct = np.nan
            
            rt = (amp_90p-amp_10p)/popt[0]
            
            corrected_time.append(correct)
            delay.append(pulse_delay)
            risetime.append(rt)
        print('No correcction applied in %i cases' %fail_counts )
        self.df['CFD_corrected_time'] = corrected_time
        self.df['pulse_delay'] = delay
        self.df['rise_time'] = risetime

    

