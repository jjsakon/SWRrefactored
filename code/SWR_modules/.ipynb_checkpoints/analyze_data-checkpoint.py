from load_data import * 
import sys
import warnings
import matplotlib.pyplot as plt
sys.path.append('/home1/efeghhi/ripple_memory/')
from SWRmodule import *
import mne
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                        MFG_labels, IFG_labels, nonHPC_MTL_labels, ENTPHC_labels, AMY_labels


def create_ripple_exists(dd_trials, ripple_start, ripple_end, min_ripple_time):
    
    ripple_exists_idxs = np.argwhere(np.sum(dd_trials['ripple'][:, ripple_start:ripple_end],axis=1)>min_ripple_time)
    ripple_exists = np.zeros(dd_trials['ripple'].shape[0])
    ripple_exists[ripple_exists_idxs] = 1
    
    return ripple_exists
    
def format_ripples(data_dict, ripple_start, ripple_end, sr, start_time, 
                         end_time):
    
    '''
    Inputs: 
    
    :param dict data_dict: 
    :param int ripple_start: start time (ms) relative to recording start time for ripple analysis
    :param int ripple_end: end time (ms) relative to recording start time for ripple analysis
    
    Outputs:
    ripple_exists: contains 1 if any electrode has a ripple, 0 else
    ripple_avg: avg number of ripples across electrodes
    '''
    
    ripple_exists_all_elecs_list = []
    
    # convert times to indices
    sr_factor = 1000/sr
    ripple_start_idx = int((ripple_start - start_time) / sr_factor)
    ripple_end_idx = int((ripple_end - start_time) / sr_factor)
    
    # list of length num_session
    # each element in the list is of shape num_trials x num_elecs x num_timesteps
    # where num_timesteps is 1500 b/c sr is 500 Hz and recording is 3000 ms
        
    for ripple_sess in data_dict['ripple']:
        ripple_sess_2d = np.reshape(ripple_sess, (-1, ripple_sess.shape[-1]))
        ripple_exists_sess = np.zeros(ripple_sess_2d.shape[0])
        ripple_exists_idxs = np.squeeze(np.argwhere(np.sum(ripple_sess_2d[:, ripple_start_idx:ripple_end_idx], axis=(1)) > 0))
        ripple_exists_sess[ripple_exists_idxs] = 1
        ripple_exists_all_elecs_list.append(ripple_exists_sess)
    
    return np.hstack(ripple_exists_all_elecs_list)

def ravel_HFA(data_dict, HFA_start=400, HFA_end=1100, sr=50, start_time=-700, 
                         end_time=2300):
    
    # convert times to indices
    sr_factor = 1000/sr
    HFA_start_idx = int((HFA_start - start_time) / sr_factor)
    HFA_end_idx = int((HFA_end - start_time) / sr_factor)
    HFA = []
    
    # average HFA time
    for HFA_sess in data_dict['HFA']:
        # reshape so that it's 2d. 
        HFA_sess_2d = np.reshape(HFA_sess, (-1, HFA_sess.shape[-1]))
        HFA.append(np.mean(HFA_sess_2d[:, HFA_start_idx:HFA_end_idx], axis=(1)))

    return np.hstack(HFA)

def create_semantic_clustered_array(data_dict, encoding_mode):
            
    '''
    :param dict data_dict: dictionary which needs to have the following keys ->
    
        clust: indicates what recalls count as clustered. There are four possible recalls:
            1) 'A': adjacent semantic
            2) 'C': remote semantic
            3) 'D': remote unclustered
            4) 'Z': dead end 
            
        position: position that each word was recalled in 
    
    :param int encoding_mode: 1 if encoding, 0 if recall 
        
    :param list clustered: which entries count as clustered
    :param list unclustered: which entries count as unclustered 
    
        The default is to use A and C as clustered, and D and Z as unclustered. 
        
    Modifies clust key to be 1 for clustered, 0 for unclustered, and -1 for everything else
    '''
    
    list_length = 12        
    semantic_array_all = data_dict['clust']
    
    clustered_all_list = []
    
    # if recalled data
    if encoding_mode == 0:
        
        # 2d array containing recall type for each item
        # second axis repeats these values across electrodes
        for sess_clust in semantic_array_all:
            
            clustered_sess_list = []
            
            semantic_array_np = sess_clust[:, 0]
            num_elecs = sess_clust.shape[1]
        
            for s in semantic_array_np:
                if s == 'A': 
                    clustered_sess_list.append(3)
                elif s == 'B':
                    clustered_sess_list.append(1)
                elif s == 'C':
                    clustered_sess_list.append(2)
                elif s == 'D':
                    clustered_sess_list.append(-1)
                elif s == 'Z':
                    clustered_sess_list.append(-2)
                else:
                    clustered_sess_list.append(0)
            
            # reshape so that its num_trials x elecs again 
            clustered_np = np.expand_dims(np.array(clustered_sess_list),axis=-1)
            clustered_np = np.repeat(clustered_np, num_elecs, axis=-1)
            clustered_all_list.append(clustered_np)
    
    # if encoding data
    else:    
        
        # get postion of recall
        recall_position_all = data_dict['position']
        
        for sess_pos, sess_clust in zip(recall_position_all, semantic_array_all):
            
            # store so that we can repeat clustered_np num_elecs times
            num_elecs = sess_pos.shape[1]
            
            clustered_sess_list = []
            recall_position_np = sess_pos[:, 0]
            semantic_array_np = sess_clust[:, 0]
            
            num_selected_trials = recall_position_np.shape[0]
                    
            for list_idx in range(0, num_selected_trials, list_length):
                
                # recall_position_np and semantic_array_np contain information 
                # about the word that was recalled and its clustering type, respectively
                # this information is repeated list_length times, so our for loop will 
                # increment by list length 
                recalled_idx = recall_position_np[list_idx] 
                cluster = semantic_array_np[list_idx]
                
                # init values to -1 so that non recalled items are -1 
                cluster_trial = [0 for x in range(list_length)]
                
                for r, c in zip(recalled_idx, cluster):
                    if r > 0 and r <= list_length:
                        if c == 'A':
                            cluster_trial[r-1] = 3 # semantic and temporal 
                        if c == 'B':
                            cluster_trial[r-1] = 1 # temporal
                        if c == 'C':
                            cluster_trial[r-1] = 2 # semantic 
                        if c == 'D':
                            cluster_trial[r-1] = -1 # neither
                        if c == 'Z': 
                            cluster_trial[r-1] = -2 # dead end
          
                
                clustered_sess_list.extend(cluster_trial)
            
            clustered_np = np.expand_dims(np.array(clustered_sess_list),axis=-1)
            clustered_np = np.repeat(clustered_np, num_elecs, axis=-1)
            clustered_all_list.append(clustered_np)
            
    return clustered_all_list


def combine_data(data_dict, **kwargs):
    
    for key, val in kwargs.items():
        data_dict[key] = val
        
    return data_dict
    
def remove_non_binary_clust(data_dict):
    
    clustered = data_dict['clust']
    mask_idxs = np.argwhere(clustered==-1)
    
    for key, val in data_dict.items():
        data_dict[key] = np.delete(val, mask_idxs, axis=0)
        
    return data_dict

def average_hfa_across_elecs(data_dict, HFA_start=400, HFA_end=1100, sr=50, start_time=-700, 
                         end_time=2300):
    
    # convert times to indices
    sr_factor = 1000/sr
    HFA_start_idx = int((HFA_start - start_time) / sr_factor)
    HFA_end_idx = int((HFA_end - start_time) / sr_factor)
    HFA = []
    
    for HFA_sess in data_dict['HFA']:
        # average HFA across electrodes and time
        HFA.append(np.mean(HFA_sess[:, :, HFA_start_idx:HFA_end_idx], axis=(1,2)))

    return np.hstack(HFA)


def getMixedEffectMeanSEs(data_dict, power, indices):
    # take a binned array of ripples and find the mixed effect SEs at each bin
    # note that output is the net ± distance from mean
    import statsmodels.formula.api as smf
    # now, to set up ME regression, append each time_bin to bottom and duplicate
    mean_values = []
    SEs = [] #CIs = []
    indices = np.squeeze(indices)
    power_array = np.squeeze(data_dict[power][indices])
    power_array_timesteps = int(power_array.shape[1])
    session_name_array = data_dict['sess'][indices]
    subject_name_array = data_dict['subj'][indices]
    
    for timestep in range(power_array.shape[1]):
        pow_binned = np.squeeze(power_array[:, timestep])
        SE_df = pd.DataFrame(data={'session':session_name_array,'subject':subject_name_array,'pow_binned': pow_binned})
        # now get the SEs JUST for this time bin
        vc = {'session':'0+session'}
        get_bin_CI_model = smf.mixedlm("pow_binned ~ 1", SE_df, groups="subject", vc_formula=vc)
        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)
        mean_values.append(bin_model.params.Intercept)
        SEs.append(bin_model.bse_fe)
        
    # get SE distances at each bin
    SE_plot = superVstack(np.array(SEs).T,np.array(SEs).T)
    
    return mean_values, SE_plot

def clust_ripple_idxs(data_dict, mode):
    
    '''
    :param dict data_dict: contains ripple_exists and clust_int key
    :param int mode:
        0 -> ripple/no ripple indices
        1 -> clust/no clust indices
        2 -> clust/no clust (only during ripples) indices
        3 -> clust/no clust (only during no ripple) indices
    '''
    clust = np.argwhere(data_dict['clust_int']==1)
    ripple = np.argwhere(data_dict['ripple_exists']==1)
    no_clust = np.argwhere(data_dict['clust_int']==0)
    no_ripple = np.argwhere(data_dict['ripple_exists']==0)
    clust_ripple = np.intersect1d(clust, ripple)
    no_clust_ripple = np.intersect1d(no_clust, ripple)
    clust_no_ripple = np.intersect1d(clust, no_ripple)
    no_clust_no_ripple = np.intersect1d(no_clust, no_ripple)
    
    # returns ripple/no ripple
    if mode==0:
        return ripple, no_ripple, "Ripple", "No ripple", "Ripple"
     # all clustered
    if mode==1:
        return clust, no_clust, "Clustered", "Not clustered", "Clust"
    # only clustered during ripple 
    if mode==2:
        return clust_ripple, no_clust_ripple, "Clustered", "Not clustered", "Clust_ripple"
    # only clustered during no ripple 
    if mode==3:
        return clust_no_ripple, no_clust_no_ripple, "Clustered", "Not clustered", "Clust_no_ripple"
    
def correct_ripple_idxs(data_dict, mode):

    correct = np.argwhere(data_dict['correct']==1)
    ripple = np.argwhere(data_dict['ripple_exists']==1)
    no_correct = np.argwhere(data_dict['correct']==0)
    no_ripple = np.argwhere(data_dict['ripple_exists']==0)
    correct_ripple = np.intersect1d(correct, ripple)
    no_correct_ripple = np.intersect1d(no_correct, ripple)
    correct_no_ripple = np.intersect1d(correct, no_ripple)
    no_correct_no_ripple = np.intersect1d(no_correct, no_ripple)
    
    # returns ripple/no ripple
    if mode==0:
        return ripple, no_ripple, "Ripple", "No ripple", "Ripple"
     # all clustered
    if mode==1:
        return correct, no_correct, "Recalled", "Not recalled", "Recalled"
    # only clustered during ripple 
    if mode==2:
        return correct_ripple, no_correct_ripple, "Recalled", "Not recalled", "Correct_ripple"
    # only clustered during no ripple 
    if mode==3:
        return correct_no_ripple, no_correct_no_ripple, "Recalled", "Not recalled", "Correct_no_ripple"
        
def plot_SCE_SME(data_dict, power, mode, region, xr, encoding_mode, behav_key, idxs1=None, 
                 idxs2=None, legend1=None, legend2=None, saveName=None, savePath=None, 
                 ymin=None, ymax=None, smoothing_triangle=5):

    power_arr = data_dict[power]
    
    if behav_key == 'correct':
        idxs1, idxs2, legend1, legend2, saveName = correct_ripple_idxs(data_dict, mode)
        savePATH = "/home1/efeghhi/ripple_memory/figures/SMEs/"
    if behav_key == 'clust':
        idxs1, idxs2, legend1, legend2, saveName = clust_ripple_idxs(data_dict, mode)
        savePATH = "/home1/efeghhi/ripple_memory/figures/clustering/"
    if behav_key == None:
        pass
        
    
    legend1 = f'{legend1}: {idxs1.shape[0]}'
    legend2 = f'{legend2}: {idxs2.shape[0]}'

    PSTH1, SE1_plot = getMixedEffectMeanSEs(data_dict, power, idxs1)
    PSTH2, SE2_plot = getMixedEffectMeanSEs(data_dict, power, idxs2)
    
    PSTH1 = triangleSmooth(PSTH1, smoothing_triangle=smoothing_triangle)
    PSTH2 = triangleSmooth(PSTH2, smoothing_triangle=smoothing_triangle)
    
    plt.plot(xr, PSTH1, label=legend1)
    plt.fill_between(xr, PSTH1-SE1_plot[0,:], PSTH1+SE1_plot[0,:], alpha = 0.3)
    plt.plot(xr, PSTH2, label=legend2)
    plt.fill_between(xr, PSTH2-SE2_plot[0,:], PSTH2+SE2_plot[0,:], alpha = 0.3)
    
    if ymin is None or ymax is None:
        pass
    else:
        plt.ylim(ymin, ymax)
        
    plt.axvline(0, color='black')
    
    plt.legend()
    
    plt.ylabel(f"{power} z-score", fontsize=14)
    
    if encoding_mode:
        plt.xlabel(f"Time from word presentation (s)", fontsize=14)
    else:
        plt.xlabel(f"Time from word retrieval (s)", fontsize=14)
        
    if encoding_mode:
        folder_save = "encoding"
    else:
        folder_save = "recall"
        
    plt.savefig(f"{savePATH}{folder_save}/{saveName}_{region}_{power}", dpi=400, bbox_inches='tight')
    
    plt.close()
    
def downsample_power(data_dict, downsample_factor=10, downsample_keys=['HFA', 'theta']):
    
    for key in downsample_keys:
        data_dict[key] = mne.filter.resample(data_dict[key], down=downsample_factor)
    
    return data_dict 
    
    

        
    
        
    

    
    
    