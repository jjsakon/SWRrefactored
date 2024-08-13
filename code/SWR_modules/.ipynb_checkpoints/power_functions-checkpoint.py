import numpy as np
import sys 
base = '/home1/john/'
sys.path.append('/home1/john/SWRrefactored/code/SWR_modules/')
from load_data_numpy import load_data_np
from comodulogram import remove_session_string

def load_z_scored_power(dd_trials, freq_range_str_arr, encoding_mode,fs, high_fq_range,low_fq_range, start_cutoff, end_cutoff): # this takes you through the filtering and z-scoring process
    
    run_Morlet = 0
        
    raw_data = dd_trials['raw']
    
    # z scoring is done for each unique subject, session, electrode combo
    subj_elec_sess_labels = dd_trials['elec_labels'] 

    for freq_range_str in freq_range_str_arr:
        
        if freq_range_str == 'high':
            freq_ranges = high_fq_range
            ylabel = 'Gamma power'
            bandwidth='auto'
        elif freq_range_str == 'low':
            freq_ranges = low_fq_range
            ylabel = 'Theta power'
            bandwidth='auto'

        # filter signal and get powers
        
        if run_Morlet == 1:
            num_freqs = 10 # of log-spaced freqs between freq_range...should make sure they don't occur at 60/120
            filtered_sig = get_filtered_signal_Morlet(raw_data, freq_ranges, start_cutoff, 
                                           end_cutoff, fs, width=5, num_freqs=num_freqs)
        else:
            # filter signal using hilbert method
            filtered_sig = get_filtered_signal_Hilbert(raw_data, freq_ranges, start_cutoff, 
                                           end_cutoff, fs, bandwidth=bandwidth)
                            
        # obtain power and amplitude
        filtered_sig = np.abs(filtered_sig) # np.real(np.abs(filtered_sig))
        filtered_sig_power = filtered_sig**2
        
        # z-score data from each electrode 
        power_z = process_power(filtered_sig_power, subj_elec_sess_labels, run_Morlet)
        power_z = power_z.squeeze() # I don't think this is necessary? Ebrahim had it here and doesn't hurt tho
        
        yield power_z, ylabel

def process_power(power, subj_elec_sess_labels, run_Morlet):
    
    from scipy.signal import decimate
      
    power = decimate(np.log10(power), 10) # decimate time

    if run_Morlet == 1:     
        # freq_range X trials X freqs X time for Morlet (e.g. 2x300x10x50)
        power = power.transpose(0, 2, 1, 3) # switch to e.g. 2x10x300x50
        power_ds_zscored = np.zeros_like(power)
        for val in np.unique(subj_elec_sess_labels): # for each elec
            val_idxs = np.argwhere(subj_elec_sess_labels==val).squeeze()
            for test in range(power.shape[0]): # for as many freq_ranges as we tested
                elec_power = power[test, :, val_idxs].squeeze() # after this a trial X freq X time_bin (e.g. 300x10x50) despite earlier transpose  
                elec_power = elec_power.transpose(1, 0, 2) # back to 10x300x50
                mean_elec_power = np.mean(elec_power, axis=(1,2), keepdims=True) # average over trials X time
                std_elec_power = np.std(np.mean(elec_power, axis=2), axis=1, keepdims=True) # average over time THEN take std over trials
                power_ds_zscored[test, :, val_idxs, :] = ((elec_power - mean_elec_power) / std_elec_power[:, np.newaxis]).transpose(1, 0, 2) # some weird indexing here but chatGPT figured it out (dims look right after the z-score but they don't slice correctly)
        
        power_ds_zscored = np.mean(power_ds_zscored,1) # after zscoring average across the 10 freqs
                
    else:       
        # input is freq_range X trials X time for Hilbert       
        power_ds_zscored = np.zeros_like(power)    
        for val in np.unique(subj_elec_sess_labels): # for each elec
            val_idxs = np.argwhere(subj_elec_sess_labels==val).squeeze()
            for test in range(power.shape[0]):
                power_ds_zscored[test, val_idxs] = z_score(power[test, val_idxs].squeeze())
        
    return power_ds_zscored
        
def get_filtered_signal_Morlet(raw_data, freq_ranges, start_idx, end_idx, fs, width=5, num_freqs=10):
    '''
    :param ndarray raw_data: n_epochs x n_timepoints
    :param list freq_range: list of list of freq ranges
    :param int start_idx, end_idx: after filtering, take only timepoints between start and end_idx
    :param int fs: sampling frequency 
    :param int width: width of the Morlet wavelet
    
    Returns the Morlet wavelet transformed signal. 
    '''     
    from mne.time_frequency import tfr_array_morlet
    
    n_trials = raw_data.shape[0]
    n_points = raw_data.shape[1]
    
    # Using tfr_array_morlet to get the time-frequency representation
    # freqs needs to be a list of center frequencies
    # data should be in the shape (n_epochs, n_channels, n_times) 
    
    filtered = np.zeros((len(freq_ranges), n_trials, num_freqs, n_points),
                    dtype=np.complex128)
    for jj, freq_limits in enumerate(freq_ranges):
        
        # get log-spaced freqs first
        freqs = np.logspace(np.log10(freq_limits[0]),np.log10(freq_limits[1]),num_freqs)
        filtered[jj,:] = np.squeeze(tfr_array_morlet(raw_data[:, np.newaxis, :],\
                                                     sfreq=fs, freqs=freqs, n_cycles=width,output='complex'))
    return filtered[:, :, :, start_idx:end_idx] # outputs freq_range X trials X freqs X time
                                    
def get_filtered_signal_Hilbert(raw_data, freq_range, start_idx, end_idx, fs, bandwidth='auto'):
    '''
    :param ndarray raw_data: n_epochs x n_timepoints
    :param list freq_range: list of lists, where each sublist is a frequency range
    :param int start_idx, end_idx: after filtering, take only timepoints between start and end_idx
    :param int fs: sampling frequency 
    
    Returns Hilbert transformed signal after bandpassing. 
    '''
    
    from mne.filter import filter_data
    from scipy.signal import hilbert
    
    n_frequencies = len(freq_range)
    n_epochs = raw_data.shape[0]
    n_points = raw_data.shape[1]
    
    filtered = np.zeros((n_frequencies, n_epochs, n_points),
                        dtype=np.complex128)    
    for jj, freq_limits in enumerate(freq_range):
        
        bandpassed_data = filter_data(raw_data, sfreq=fs, l_freq=freq_limits[0], 
                       h_freq=freq_limits[1], l_trans_bandwidth=bandwidth, 
                       h_trans_bandwidth=bandwidth, verbose=False)
        filtered[jj, :] = hilbert(bandpassed_data) # outputs freq_range X trials X time
    return filtered[:, :, start_idx:end_idx] # idx in samples
    
def z_score(power):
    
    '''
    :param ndarray power: 2d array of shape trials x timesteps (only for Hilbert...which doesn't go over freqs like Morlet)
    '''
    
    '''
        # at the HFA_morlet step in Sakon code the matrix is:
            freqs X words X elecs X time_bins (e.g. 10x300x108x30)

        # z-score using std of time bin averaged instead (mean is same either way) after talking to Mike 2022-03-08
        HFA_morlet = (HFA_morlet - np.mean(HFA_morlet, axis=(1,3))) / np.std(np.mean(HFA_morlet, axis=3),axis=1)
        HFA_morlet = np.mean(HFA_morlet,0) # mean over the 10 frequencies (now down to events X pairs X 100 ms bins)
        
        # So what this does is subtract the average across trials and time_bins then divides by std across trials
        # after averaging over time bins. THEN you avg acrsos freqs to end up with trials X elecs X time_bins
    '''
    
    # mean center by mean across time and trials 
    # then divide by standard deviation of the average across timesteps
    power = (power - np.mean(power)) / np.std(np.mean(power, axis=1),axis=0)
    return power      