import sys
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import * 
from analyze_data import * 
import warnings
import matplotlib.pyplot as plt
from SWRmodule import *
sys.path.append('/home1/efeghhi/ripple_memory/')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                        MFG_labels, IFG_labels, nonHPC_MTL_labels, ENTPHC_labels, AMY_labels
                        
'''
Creates SCE and SME figures for specified frequency band.
'''
############### set parameters ###############
encoding_mode = 1
power_bands = ['low_gamma', 'high_gamma']
behav_key = 'clust'
condition_on_ca1_ripples = False

sr = 500 # sampling rate in Hz
sr_factor = 1000/sr 

if encoding_mode: 
    start_time = -700
    end_time = 2300
    num_bins = 150
    
    catFR_dir = '/scratch/efeghhi/catFR1/ENCODING/'
    
    # find ripples within these timepoints
    # this corresponds to 400ms to 1100ms post-word onset 
    ripple_start = 400
    ripple_end = 1100
    
    ymin = -0.25
    ymax = 2.0

else:
    
    start_time = -2000
    end_time = 2000
    num_bins = 200
    catFR_dir = '/scratch/efeghhi/catFR1/IRIonly/'
    
    # -600 to -100 ms (0 ms is word onset)
    ripple_start = -600
    ripple_end = -100
    
    ymin = None
    ymax = None
    
downsample_factor = 5

# convert ripple_start and ripple_end to idxs
ripple_start_idx = int((ripple_start - start_time)/sr_factor)
ripple_end_idx = int((ripple_end-start_time)/sr_factor)

num_bins = int(num_bins/downsample_factor)
xr = np.linspace(start_time/1000, end_time/1000, num_bins)
##############################################

region_name = ['HPC', 'AMY'] # if empty list, loads all data

# load all data
data_dict = load_data(catFR_dir, region_name=region_name, encoding_mode=encoding_mode, 
                      condition_on_ca1_ripples=condition_on_ca1_ripples)

if encoding_mode: 
    data_dict = remove_wrong_length_lists(data_dict)
    
# ca1
ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
data_dict_ca1 = select_region(data_dict, ca1_elecs)
count_num_trials(data_dict_ca1, "ca1")

# dg
ca3_dg_elecs = [x for x in HPC_labels if 'dg' in x]
data_dict_ca3_dg = select_region(data_dict, ca3_dg_elecs)
count_num_trials(data_dict_ca3_dg, "ca3_dg")

# amy
data_dict_amy = select_region(data_dict, AMY_labels)
count_num_trials(data_dict_amy, "amy")

# entphc
data_dict_entphc = select_region(data_dict, ENTPHC_labels)
count_num_trials(data_dict_entphc, "entphc")

data_dicts = [data_dict_ca1, data_dict_ca3_dg, data_dict_amy, data_dict_entphc]
data_dict_labels = ['ca1', 'dg', 'amy', 'entphc']

# if behav key is clust (if correct, everything is the same but replace clust with correct) -> 
# 0 is ripple/no ripple, 1 is clust/no clust, 2 is 1 but ripple only, 3 is 1 but no ripple only 
# pass in a mode for each brain region 
modes = [[3], [1,2,3], [3], [1,2,3]] 
skip_regions = ['dg', 'entphc']

for data_dict_region, brain_region, mode in zip(data_dicts, data_dict_labels, modes):

    print("Running on brain region: ", brain_region)
    if brain_region in skip_regions:
        continue
    
    # create clustered int array
    clustered_int = create_semantic_clustered_array(data_dict_region, encoding_mode)
    data_dict_region['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict_region, order='C')
    
    ripple_exists = create_ripple_exists(dd_trials, ripple_start_idx, ripple_end_idx, 0)
    dd_trials['ripple_exists'] = ripple_exists
        
    dd_trials = downsample_power(dd_trials, downsample_factor=downsample_factor, 
                                 downsample_keys=['HFA', 'theta', 'low_gamma', 'high_gamma'])
    
    for m in mode:
        
        print("Mode: ", m)
        print("Brain region: ", brain_region)
        
        for p in power_bands:
            
            plot_SCE_SME(dd_trials, power=p, mode=m, xr=xr, region=brain_region, 
                ymin=ymin, ymax=ymax, smoothing_triangle=5, encoding_mode=encoding_mode, behav_key=behav_key)
        


