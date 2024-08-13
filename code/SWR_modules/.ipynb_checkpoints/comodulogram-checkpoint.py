import numpy as np
import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import decimate, resample
from scipy.signal import hilbert


def remove_session_string(subject_session_elec_string):
    
    subj_sess = subject_session_elec_string.split('_')[0]
    subj = subj_sess.split('-')[0]
    
    elec = subject_session_elec_string.split('_')[1]
    
    subj_elec = f"{subj}_{elec}"
    
    return subj_elec

def get_filtered_signal(raw_data, freq_range, start_idx, end_idx, fs, bandwidth='auto'):
    
    '''
    :param ndarray raw_data: n_epochs x n_timepoints
    :param list freq_range: list of lists, where each sublist is a frequency range
    :param int start_idx, end_idx: after filtering, take only timepoints between start and end_idx
    :param int fs: sampling frequency 
    
    Returns hilbert transformed signal after bandpassing. 
    '''
    
    from mne.filter import filter_data
    
    n_frequencies = len(freq_range)
    n_epochs = raw_data.shape[0]
    n_points = raw_data.shape[1]
    
    filtered = np.zeros((n_frequencies, n_epochs, n_points),
                        dtype=np.complex128)
    
    for jj, frequency in enumerate(freq_range):
        
        bandpassed_data = filter_data(raw_data, sfreq=fs, l_freq=frequency[0], 
                       h_freq=frequency[1], l_trans_bandwidth=bandwidth, 
                       h_trans_bandwidth=bandwidth, verbose=False)
        filtered[jj, :] = hilbert(bandpassed_data)
    return filtered[:, :, start_idx:end_idx] # idx in samples

def compute_pac(low_sig, high_sig, n_bins=18, min_shift=50, n_surrogates=300):
    
    
    n_low = low_sig.shape[0]
    n_high = high_sig.shape[0]
    
    phase_vals = np.real(np.angle(low_sig))
    amplitude_vals = np.real(np.abs(high_sig))

    # add eps so that pi is included in a bin
    eps=1e-4
    phase_bins = np.linspace(-np.pi, np.pi+eps, n_bins+1)
    # get the indices of the bins to which each value in input belongs (0-18)
    phase_preprocessed = np.digitize(phase_vals, phase_bins) - 1
    
    MI_vals = []
    phase_amp_hist_storage = []
    
    # for each low/high frequency pair, compute MI and phase amplitude distribution
    for low in range(n_low):
        
        phase = phase_preprocessed[low]
        
        for high in range(n_high):
            
            amp = amplitude_vals[high]
            MI_surrogates, amplitude_dist_surrogates = modulation_index(phase, amp, n_surrogates, n_bins, min_shift)
            
            MI_vals.append(MI_surrogates)
            phase_amp_hist_storage.append(amplitude_dist_surrogates)

            
    return MI_vals, phase_amp_hist_storage

def circular_permute_within_epochs(data_arr, min_shift):
    
    n_epochs = data_arr.shape[0]
    n_points = data_arr.shape[1]
    
    data_arr_shuffled = np.zeros_like(data_arr)
    
    
    for epoch_idx in range(n_epochs):
        
        # circular permute each epoch
        shift_val = np.random.randint(min_shift, n_points-min_shift)
        data_arr_shuffled[epoch_idx] = np.hstack((data_arr[epoch_idx, shift_val:], data_arr[epoch_idx, :shift_val]))
        
    return data_arr_shuffled

    
def modulation_index(phase_orig, amplitude, n_surrogates, n_bins, min_shift):
    
    from copy import deepcopy
    
    MI_surrogates = []
    amplitude_dist_surrogates = []
    
    for n in range(n_surrogates+1):
         
        if n > 0:
            phase = circular_permute_within_epochs(phase_orig, min_shift)
        else:
            phase = phase_orig
 
        phase_1d = np.ravel(phase)
        amplitude_1d = np.ravel(amplitude)
 
        amplitude_dist = np.ones(n_bins)  # default is 1 to avoid log(0)
    
        # average across all gamma values that occur at a given theta phase bin
        for b in np.unique(phase_1d):
            selection = amplitude_1d[phase_1d == b]
            amplitude_dist[b] = np.mean(selection)

        # Kullback-Leibler divergence of the distribution vs uniform
        amplitude_dist /= np.sum(amplitude_dist) # make into a pdist
        divergence_kl = np.sum(
            amplitude_dist * np.log(amplitude_dist * n_bins))
        
        # divide by log(n_bins) to bound by 1
        MI = divergence_kl / np.log(n_bins)
            
        MI_surrogates.append(MI) 
        
        # the first value is without shuffling
        amplitude_dist_surrogates.append(amplitude_dist)
       
    
    return MI_surrogates, amplitude_dist_surrogates


def save_MI_amplitude(subj_elec_idxs_all, raw_data, postfix_save, fs, savePath, 
                      low_fq_range, high_fq_range, start_idx, end_idx, 
                      subj_elec_min_epochs_condition, match_trial_count=True, bandwidth_lfreq=2, 
                      bandwidth_hfreq=18, random_seed=42):
    
    '''
    :param ndarray subj_elec_idxs_all: array of shape num_epochs, 
            which indicates which subject+electrode combo that epoch belongs to
            
    :param ndarray raw_data: raw neural data, num_epochs x num_timepoints
    
    :param str postfix_save: MI and PAC distributions are saved as subj_elec_postfix_save
    
    :param int fs: sampling frequency
    
    :param str savePath: folder to save data
    '''
    
    np.random.seed(random_seed)

    for subj_elec in np.unique(subj_elec_idxs_all):
        
        save_dict = {}

        subj_elec_idxs = np.argwhere(subj_elec_idxs_all==subj_elec).squeeze()
        
        num_epochs = subj_elec_idxs.shape[0] 
        
        if match_trial_count:

            if subj_elec in subj_elec_min_epochs_condition.keys():

                min_epochs = int(subj_elec_min_epochs_condition[subj_elec])

                # randomly select a subportion of epochs so that trial count
                # is equivalent across conditions
                if min_epochs < num_epochs:
                    print(min_epochs, num_epochs)
                    selected_epochs = np.random.randint(0, num_epochs, min_epochs)
                    subj_elec_idxs = subj_elec_idxs[selected_epochs].squeeze()
            else:
                continue

        if subj_elec_idxs.shape[0] < 10:
            continue

        raw_data_elec_subj = raw_data[subj_elec_idxs].squeeze()

        low_sig = get_filtered_signal(raw_data_elec_subj, low_fq_range, start_idx, end_idx, fs=fs, bandwidth=bandwidth_lfreq)
        high_sig = get_filtered_signal(raw_data_elec_subj, high_fq_range, start_idx, end_idx, fs=fs, bandwidth=bandwidth_hfreq)
 
        MI, pac_hist = compute_pac(low_sig, high_sig, n_bins=18, min_shift=50, n_surrogates=300)

        save_dict['MI'] = MI
        save_dict['pac_hist'] = pac_hist

        np.savez(f'{savePath}{subj_elec}_{postfix_save}', **save_dict)
        
        print("SAVED DATA")

