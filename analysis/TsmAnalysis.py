import pandas as pd
import numpy as np

from tqdm import tqdm

import warnings

from scipy.optimize import curve_fit

    
class TsmAnalysis:

    
    """
    Class for processing data acquired with the time structure monitors at HIPA.
    TODO Write the docs!

    Attributes:
        compute:
        rf_model:
        falling_edge_model:
        phase_to_time:
        optimizer:
        get_rf_timing:
        chi_square_test:
        CFD_correction:
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

        df = pd.DataFrame()
        df['time'] = df_sig_in['time']
        # df['voltage'] = df_sig_in['voltage']
        df['trigger_timing'] = df_sig_in['trigger_timing']
        # df['bsl'] = df_sig_in['voltage'].apply(lambda v: np.median(v[bsl_window[0]:bsl_window[1]]))
        df['rms'] = df_sig_in['voltage'].apply(lambda v: np.sqrt(np.mean(np.square(v[:len(v)//3]))))
        # df['pmt'] = df.apply(lambda v: list(np.array(v.voltage) - v.bsl), axis=1)
        df['pmt'] = df_sig_in['voltage']
        df['amp'] = df['pmt'].apply(lambda v: np.min(v))
        df['min_loc'] = df['pmt'].apply(lambda v: np.where(v == np.min(v))[0][0])
        df['ref'] = df_ref_in['voltage']
        df['area'] = df['pmt'].apply(lambda v: np.sum(-np.array(v)[bsl_window[1]:]) )

        pulse_start = []
        for i in tqdm(df.index):
            time_range = np.abs(df.time[i])
            time_zero = np.min(time_range)
            pulse_start.append(time_zero)

        df['pulse_start'] = pulse_start
               
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
       
        
    def optimizer(self, model, time_arr, sig_arr, init_params=None,
                   bounds=None):
        """
        Wrapper for curve_fit plus GoF test
        """
        try:
            popt, pcov = curve_fit(model, time_arr, sig_arr, p0=init_params, bounds=bounds)
            chi2, dof = self.chi_square_test(time_arr, sig_arr, popt)
            GoF = chi2/dof
        except:
            popt, pcov = np.nan, np.nan
            GoF = np.nan
        return popt, pcov, GoF
    
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
                        
            if cfd_bounds[0] > cfd_bounds[1] or cfd_bounds[0] < 0 or cfd_bounds[1] > 1 or cfd_bounds[1] < 0 or cfd_bounds[1] > 1 :
                cfd_bounds = (.1, .9)
                warnings.warn('provided cfd_bounds (%f, %f) are invalid. Make sure both are in range (0,1) and the first boundary is lower than the second.' %cfd_bounds)
            
            self.CFD_correction(lower=cfd_bounds[0], upper=cfd_bounds[1])            
            timing_col = 'CFD_corrected_time'

        elif timing == 'original':
            timing_col = 'time'
        elif timing not in ['corrected', 'original']:
            warnings.warn(f'timing identifier %s not recognized. The computation defaults to scope-native timebase and is prone to discriminator walk.' %timing)
            timing_col = 'time'
        
        print('Calculating pulse timing w.r.t. RF')

        ref_t = []
        gof = []

        rf_amp = []
        rf_freq = []
        rf_phase = []
        rf_bsl_shift = []

        fit_fail_counter = 0

        init_params = [1.3, 0, 2*np.pi*50.6e6, 0]
        bounds = ([1.26, -1e-2, 3.1e8, -np.inf], [1.3, 1e-2, 3.1825e8, np.inf])

        for i in tqdm(self.df.index):
            
            time_arr = np.array(self.df[timing_col][i])
            ref_arr = np.array(self.df.ref[i])

            mask = (time_arr > -1e-8) & (time_arr < 1e8) # Perfrom fit on 20ns window around origin
            
            try:
                time_arr = time_arr[mask]
                ref_arr = ref_arr[mask]
                
                init_params = [1.3, 0, 2*np.pi*50.6e6, 0]
                bounds = ([1.26, -1e-2, 3.1e8, -np.inf], [1.3, 1e-2, 3.1825e8, np.inf])

            
                popt, pcov, GoF = self.optimizer(self.rf_model, time_arr, ref_arr,
                                                 init_params=init_params, bounds=bounds)

                amp, bsl_shift, ang_freq, phase = popt
                freq = ang_freq/2/np.pi
               
            except:
                fit_fail_counter += 1
                amp, bsl_shift, freq, phase, GoF = np.nan, np.nan, np.nan, np.nan, np.nan
    
            dt = self.phase_to_time(phase, freq)

            ref_t.append(dt)
            gof.append(GoF)
            
            rf_amp.append(amp)
            rf_freq.append(freq)
            rf_phase.append(phase)
            rf_bsl_shift.append(bsl_shift)

        self.df['ref_timing'] = -1*np.array(ref_t)
        
        self.df['rf_amp'] = rf_amp
        self.df['rf_freq'] = rf_freq
        self.df['rf_phase'] = rf_phase
        self.df['rf_bsl_shift'] = rf_bsl_shift

        self.df['GoF'] = gof
        self.df['fit_misalignment'] = self.df.apply(lambda v: self.phase_to_time(v.GoF, v.rf_freq), axis=1)

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
            dof:  Integer - Number of degrees of freedom of the model.
        """
        residuals = y - self.rf_model(x, *popt)
        chi_square = np.sum(residuals ** 2)
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

        print('Estimating pulse arival times')
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

                popt, pcov, = curve_fit(self.falling_edge_model, time, amp, p0=p0, bounds=bounds)
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
        print('No corresction applied in %i cases' %fail_counts )
        self.df['CFD_corrected_time'] = corrected_time
        self.df['pulse_delay'] = delay
        self.df['rise_time'] = risetime

    

