import numpy as np

import sys 
sys.path.append('/home1/john/SWRrefactored/code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import resample

from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert


def load_data_np(encoding_mode, task, region_name=['HPC'], subregion=['ca1'], train_only=True, 
                    base_scratch = '/scratch/john/SWRrefactored/patient_info/'):

    print("Loading data")
    
    condition_on_ca1_ripples = False
    
    if encoding_mode: 
        if task=='catFR1': 
            files_dir = f'{base_scratch}catFR1/ENCODING/'
        elif task == 'FR1':
            files_dir = f'{base_scratch}FR1/ENCODING/'
    else:
        if task=='catFR1':
            files_dir = f'{base_scratch}catFR1/IRIonly/'
        elif task == 'FR1':
            files_dir = f'{base_scratch}FR1/IRIonly/'
        
    print(f"LOADING DATA FROM: {region_name[0]} FOR EXPERIMENT {task}")
        
    data_dict, one_d_keys = load_data(directory=files_dir, region_name=region_name, 
                          encoding_mode=encoding_mode, train_only=train_only)
    
    if task == False:
        data_dict.pop('clust')
        data_dict.pop('category_array')

    if encoding_mode: 
        data_dict = remove_wrong_length_lists(data_dict, one_d_keys)
    
    # leave empty to select all electrodes
    selected_elecs = []
    if region_name == ['HPC']:
        if subregion == ['']: # take all HPC elecs
            selected_elecs = [x for x in HPC_labels]
        else:
            for s in subregion:
                selected_elecs_s = [x for x in HPC_labels if s in x]
                selected_elecs.extend(selected_elecs_s)    
    elif region_name == ['ENT']:
        selected_elecs = [x for x in ENTPHC_labels]
    elif region_name == ['AMY']:
        selected_elecs = [x for x in AMY_labels]
    else:
        print('No region by that name in these parts')
        error
 
    if len(selected_elecs) > 0:
        data_dict = select_region(data_dict, selected_elecs, one_d_keys)

    # create clustered int array
    if task:
        clustered_int = create_semantic_clustered_array(data_dict, encoding_mode)
        data_dict['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict, order='C')
    
    return dd_trials


