# I use getSubDFforSlurm.ipynb instead now

import pandas as pd; pd.set_option('display.max_columns', 30)
import numpy as np
from cmlreaders import CMLReader, get_data_index
import ptsa
import sys
import os
import matplotlib.pyplot as plt
from pylab import *
from copy import copy
from scipy import stats
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
from general import *
from SWRmodule import *
import csv
import os
import dill, pickle
import mne
from copy import copy    
from scipy.signal import firwin,filtfilt,kaiserord
from ptsa.data.filters import ButterworthFilter, ResampleFilter, MorletWaveletFilter
import xarray as xarray
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                        MFG_labels, IFG_labels, nonHPC_MTL_labels, ENTPHC_labels, AMY_labels
# from SWRmodule import CMLReadDFRow,get_bp_tal_struct,get_elec_regions,ptsa_to_mne,detectRipplesHamming
# from general import superVstack,findInd,findAinB
# from SWRmodule import downsampleBinary,LogDFExceptionLine,getBadChannels,getStartEndArrays,getSecondRecalls,\
#                     removeRepeatedRecalls,getSWRpathInfo,selectRecallType,correctEEGoffset,\
#                     getSerialposOfRecalls,getElectrodeRanges,\
#                     detectRipplesHamming,detectRipplesButter,getRetrievalStartAlignmentCorrection,\
#                     removeRepeatsBySerialpos,get_recall_clustering, compute_morlet # specific to clustering

import pingouin as pg
import argparse

'''
This script loads all file names for a given experiment. 
It then iterates through these files and checks whether the data
corresponding to those files have a) already processed and stored
within the user-specified save_path, or b) are in the excluded 
sessions list. If neither criteria is met, the file is placed in
a temp pickle file, which create_events_mne.py uses to process 
the data.

RERUN THIS FILE BEFORE RUNNING CREATE_EVENTS.SLURM. If you do not, 
the temp pickle file won't be updated. 
'''


available_regions = [HPC_labels, ENTPHC_labels, AMY_labels,temporal_lobe_labels,MFG_labels,
                     IFG_labels,nonHPC_MTL_labels,ENT_labels,PHC_labels]

################################################################
df = get_data_index("r1") # all RAM subjects
exp = 'FR1' # 'catFR1' #'FR1'
save_path = f'/scratch/john/SWRrefactored/patient_info/{exp}/'
### params that clusterRun used
selected_period = 'encoding' # surrounding_recall # whole_retrieval # encoding 
recall_type_switch = 10 # 0 for original, 1 for only those with subsequent, 2 for second recalls only, 3 for isolated recalls
brain_region_idxs = np.arange(9) #9
remove_soz_ictal = 0
recall_minimum = 2000
filter_type = 'hamming'
extra = '' #'-ZERO_IRI'
################################################################

def create_temp_df_func(df, exp, save_path, selected_period, recall_type_switch, 
                   selected_region, remove_soz_ictal, recall_minimum, filter_type, extra):
    
    if selected_region == HPC_labels:
        region_name = 'HPC'
    elif selected_region == ENT_labels:
        region_name = 'ENT'
    elif selected_region == PHC_labels:
        region_name = 'PHC'
    elif selected_region == temporal_lobe_labels:
        region_name = 'TEMPORALLOBE'
    elif selected_region == MFG_labels:
        region_name = 'MFG'
    elif selected_region == IFG_labels:
        region_name = 'IFG'
    elif selected_region == nonHPC_MTL_labels:
        region_name = 'nonHPC_MTL'
    elif selected_region == ENTPHC_labels:
        region_name = 'ENTPHC'
    elif selected_region == AMY_labels:
        region_name = 'AMY'

    print(f"Region name: {region_name}")

    # dataframe containing experiment session info
    exp_df = df[df.experiment==exp] 
    
#     # select subs
#     subs = ['R1054J','R1345D','R1048E','R1328E','R1308T', # first 2 are sr ≥ 1000. 3rd is 500 Hz.
#         'R1137E','R1136N','R1094T','R1122E','R1385E'] # nice example FR1 subs used in Fig. 2
#     exp_df = df[(df.subject.isin(subs))  & (df.experiment == exp)] # all sessions for subs

    # 575 FR sessions. first 18 of don't load so skip those 
    if exp == 'FR1':
        exp_df = exp_df[
                        ((df.subject!='R1015J') | (df.session!=0)) & 
                        ((df.subject!='R1063C') | (df.session!=1)) & 
                        ((df.subject!='R1093J') | (~df.session.isin([1,2]))) &
                        ((df.subject!='R1100D') | (~df.session.isin([0,1,2]))) &
                        ((df.subject!='R1120E') | (df.session!=0)) &
                        ((df.subject!='R1122E') | (df.session!=2)) &
                        ((df.subject!='R1154D') | (df.session!=0)) &
                        ((df.subject!='R1186P') | (df.session!=0)) &
                        ((df.subject!='R1201P') | (~df.session.isin([0,1]))) &
                        ((df.subject!='R1216E') | (~df.session.isin([0,1,2]))) &
                        ((df.subject!='R1277J') | (df.session!=0)) &
                        ((df.subject!='R1413D') | (df.session!=0)) & 
                        ((df.subject!='R1123C') | (df.session!=2)) & # artifacts that bleed through channels (see SWR FR1 prob sessions ppt)
                        ((df.subject!='R1151E') | (~df.session.isin([1,2]))) & # more bleed-through artifacts (see same ppt)
                        ((df.subject!='R1275D') | (df.session!=3)) & # 3rd session an actual repeat of 2nd session (Paul should have removed from database by now)
                        ((df.subject!='R1311T') | (df.session!=0)) & ## these next 3 eegoffset -1 for many recalls so messes things up for clustering analysis ##
                        ((df.subject!='R1113T') | (df.session!=0)) &
                        ((df.subject!='R1137E') | (df.session!=0)) 
                       ] 
    if exp == 'catFR1':
        exp_df = exp_df[
                        ((df.subject!='R1044J') | (df.session!=0)) & # too few trials to do pg pairwise corr
                        ((df.subject!='R1491T') | (~df.session.isin([1,3,5]))) & # too few trials to do pg pairwise corr
                        ((df.subject!='R1486J') | (~df.session.isin([4,5,6,7]))) & # repeated data...will be removed at some point... @@
                        ((df.subject!='R1501J') | (~df.session.isin([0,1,2,3,4,5]))) & # these weren't catFR1 (and they don't load right anyway)
                        ((df.subject!='R1235E') | (df.session!=0)) & # split EEG filenames error...documented on Asana
                        ((df.subject!='R1310J') | (df.session!=1)) & # session 1 is just a repeat of session 0
                        ((df.subject!='R1239E') | (df.session!=0)) # some correlated noise (can see in catFR1 problem sessions ppt)
                        ]


    exclude_sessions = []

    exclude_from_rerun = 'exclude_participants.csv'
    file_exists = os.path.exists(exclude_from_rerun)
    field_name = 'Filename'
    excluded_sessions = []

    # load excluded sessions into a list to stop from re-running them again 
    if file_exists:
        excluded_sessions_csv = pd.read_csv(exclude_from_rerun)
        excluded_sessions = np.array(list(excluded_sessions_csv.Filename))
        exclusion_reason = list(excluded_sessions_csv['Reason for failure'])

    soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,recall_type_switch,selected_period,recall_minimum)


    rerun_mask = []

    num_excluded = 0
    num_files = 0
    import ipdb; ipdb.set_trace()
    for i,row in enumerate(exp_df.itertuples()):

        num_files += 1
        sub = row.subject; session = row.session; exp = row.experiment
        path_name = os.path.join(save_path, subfolder)
        fn = os.path.join(path_name,
                'SWR_'+exp+'_'+sub+'_'+str(session)+'_'+region_name+'_'+selected_period+recall_selection_name+
                            '_'+soz_label+'_'+filter_type+extra+'.p') 

        # if file is in excluded_sessions, don't add it to the rerun_df
        if fn in excluded_sessions:
            excluded_session_idx = int(np.argwhere(fn==excluded_sessions)[0])
            exclusion_reason_fn = exclusion_reason[excluded_session_idx]

            # allow data to be generated for subjects without any ripples for now
            if exclusion_reason_fn == 'Ripple array is empty':
                pass
            else:
                num_excluded += 1
                continue 
        try:
        # if pickle file exists for session, don't add it to the rerun_df 
            with open(fn,'rb') as f:
                dat = pickle.load(f)
        except:
            rerun_mask.append(i)

    # dataframe containing all the sessions to rerun   
    rerun_df = exp_df.iloc[rerun_mask]

    # save as dill so can bypass pickling in ipython for cluster parallelization
    import dill
    # save rerun_df to pickle when using slurm,
    # slurm works by doing a N sessions in parallel, 
    # where each slurm process indexes a row of rerun_df
    # if we just pass rerun_df directly and access the first
    # row, then we won't know if another slurm process is already working
    # on that row. If each slurm process calls the next row,
    # the row index will eventually exceed the size of rerun_df because
    # rerun_df grows smaller everytime a session is saved.

    # opening a file with wb automatically clears its contents 
    print("SAVING TEMP DF")
    with open(f'/scratch/john/SWRrefactored/patient_info/temp_dfSWR_{region_name}_{selected_period}_{exp}.p', 'wb') as f: 
        dill.dump(list(rerun_df.itertuples()) ,f)

    params = []
    for i in range(len(list(rerun_df.itertuples()) )):
        params.append(i)

    print("Number of sessions left: ", len(params))
    print("Number of excluded sessions: ", num_excluded)
    print("Num files: ", num_files)
    
    
for i in brain_region_idxs:
    
    br = available_regions[i]
    
    print("Running script for: ", br)

    create_temp_df_func(df, exp, save_path, selected_period, recall_type_switch, 
                   br, remove_soz_ictal, recall_minimum, filter_type, extra)