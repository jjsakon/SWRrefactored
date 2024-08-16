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
sys.path.append('/home1/john/SWRrefactored/code/SWR_modules/')
from SWRmodule import *
from general import * #superVstack,findInd,findAinB
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
# from SWRmodule import *
# downsampleBinary,LogDFExceptionLine,getBadChannels,getStartEndArrays,getSecondRecalls,\
#                     removeRepeatedRecalls,getSWRpathInfo,selectRecallType,correctEEGoffset,\
#                     getSerialposOfRecalls,getElectrodeRanges,\
#                     detectRipplesHamming,detectRipplesButter,getRetrievalStartAlignmentCorrection,\
#                     removeRepeatsBySerialpos,get_recall_clustering, compute_morlet # specific to clustering

import pingouin as pg

################################################################
df = get_data_index("r1") # all RAM subjects
exp = 'catFR1' # 'catFR1' #'FR1'
save_path = f'/scratch/john/SWRrefactored/patient_info/{exp}/'
### params that clusterRun used
selected_period = 'surrounding_recall' # surrounding_recall # whole_retrieval # encoding 
remove_soz_ictal = 0
recall_minimum = 2000
filter_type = 'hamming'
extra = '' #'- ZERO_IRI'
available_regions = {
    "HPC_labels": HPC_labels,
    "ENTPHC_labels": ENTPHC_labels,
    "AMY_labels": AMY_labels
}
brain_region_idxs = np.arange(len(available_regions))
if selected_period == 'encoding':
    recall_type_switch = 10 # no IRI requirements for recalls that is necessary for surrounding_recall
elif selected_period == 'surrounding_recall':
    recall_type_switch = 0 # 0 for original (remove IRI < recall_minimum), 1 for only those with subsequent, 2 for second recalls only, 3 for isolated recalls
else:
    print('make sure you read through and understand what getSWRpathInfo is doing')
################################################################

def ClusterRunSWRs(row, selected_region,save_path, selected_period, 
                   exp, testing_mode=False):
    
    '''
    :param [int, str] temp_df_select: 
        if int, then temp_df_select is used to index a row of temp_df
        if str, then must be name of a patient. function iterates through 
        rows of temp_df until row.subject equals temp_df_select
        
    :param str selected_period: encoding, surrounding_recall, or whole_retrieval
    :param str selected_region: which brain region to generate data for 
    :param str save_path: folder to save data in
    :param str exp: catFR or FR
    :param bool testing_mode: if in testing mode, saves data to "test.p" 
    so as to not overwrite existing data
    '''
    
    def add_session_to_exclude_list(message):

            # add session to csv so it's not included in rerun_df
            # defining this function within ClusterRunSWRs because
            # when running using slurm, slurm doesn't have access
            # to variables outside the scope of ClsuterRunSWRs

            soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,recall_type_switch,
                                                                    selected_period,recall_minimum)
            path_name = save_path+subfolder

            fn = os.path.join(path_name,
                'SWR_'+exp+'_'+sub+'_'+str(session)+'_'+region_name+'_'+selected_period+recall_selection_name+
                            '_'+soz_label+'_'+filter_type+'.p')

            data_dict = {'Filename': fn, 'Reason for failure': message}

            file_path = '/scratch/john/SWRrefactored/patient_info/exclude_participants.csv'

            file_exists = os.path.exists(file_path)

            fieldnames = data_dict.keys()

            with open(file_path, 'a', newline='') as csvfile:

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(data_dict)      
    
    
    ### PARAMS ###
    
    ##
    ## NOTE: for looking at individual lags make sure min_recalls is set to 0 or will eliminate trials from low recall lists
    ##
    
    save_values = 1
    
    # there are three periods this code is set up to look at: periods aligned to recall, the entire retrieval period, and the encoding period
      # switch to determine which recalls to look at! (see details below)
    # (as of 2021 always leave this as 0, since I select for 4/6/etc below)
    # 0: Original analysis taking only recalls without a recall in 2 s IRI before them
    # 1: Take these same recalls, but keep only those WITH a recall within 2 s after they occur 
    # 2: test condition where we look at second recalls within IRI ONLY (there is an initial recall in 2 s before current recall)
    # 3: isolated recalls with no other recalls +/- RECALL_MINIMUM s
    # 4: only first recall of every retrieval period
    # 5: take only those recalls that come second in retrieval period within 2 s of first retrieval
    # 6: take only NOT first recall of every retrieval period
    # 7: take only NOT first recall AND ISOLATED trials (this should REALLY maximize SWR bump)
    # 10: same as 0 but with no IRI (mostly just to see number of recalls)
    remove_soz_ictal = 0 # 0 for nothing, 1 for SOZ, 2 for SOZ+ictal    
    min_ripple_rate = 0.1 # Hz.
    max_ripple_rate = 1.5 # Hz.
    max_trial_by_trial_correlation = 0.05 # if ripples correlated more than this remove them
    max_electrode_by_electrode_correlation = 0.2

    ### semantic/temporal clustering parameters ###
    min_recalls = 0
    PCA_ndim = 1 # number of PC dims to use for semantic clustering (Ethan usually found only 1 worked for theta/FC)
    
#     # for parametric run through recall_minimums
#         recall_mins = np.arange(800,5100,100) #[800,900,1100,1200,1300,1400]
# #     recall_mins = [1600,1700,1800,1900,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
#     for recall_minimum in recall_mins:
    
    # recall params
    recall_minimum = 2000 # for isolated recalls what is minimum time for recall to be considered isolated?
    IRI = 2000 # inter-ripple interval...remove ripples within this range (keep only first one and remove those after it)
    retrieval_whole_time = 10000
    # encoding params
    encoding_time = 2300 # actual preentation is 1.6 s + 0.75-1.0 s so keep +700 ms so can plot +500 ms
    pre_encoding_time = -700 # since minimum ISI is 0.75 s let's only plot the 500 ms before word on with a 200 ms buffer
    # these aren't likely to be changed:
    desired_sample_rate = 500. # in Hz. This seems like lowerst common denominator recording freq.
    eeg_buffer = 1000 # buffer to add to either end of IRI when processing eeg #**
    filter_type = 'hamming' # see local version below for details. 'butter' is removed from cluster version so don't change this    
    ### END PARAMS ###    
    
    # get region label
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

    if selected_period == 'surrounding_recall':
        if IRI == 0:
            psth_start = -2000
            psth_end = 2000
        else:
            psth_start = -IRI # only makes sense to look at period <= IRI
            psth_end = IRI # how long to grab data after recall
    elif selected_period == 'whole_retrieval':
        psth_start = -IRI # doesn't have to be IRI just 2000 ms is convenient
        psth_end = IRI+retrieval_whole_time
    elif selected_period == 'encoding':
        psth_start = pre_encoding_time
        psth_end = encoding_time

    ripple_array = []; fr_array = []; 
    trial_nums = []; 
    elec_names = []; sub_sess_names = []; sub_names = []
    electrodes_per_session = []
    total_lists = 0; total_recalls = 0; kept_recalls = 0
    align_adjust = 0
    ent_elec_ct = []; sd_regions = []; not_sd_regions = []
    ripple_ied_accum_ct = []
    time_add_save = []
    encoded_word_key_array = []; category_array = []
    
    session_ripple_rate_by_elec = []; # for cluster version only since I compare across electrodes for each session
    
    list_recall_num_array = []; rectime_array = []; # new ones added 2020-11-24
    serialpos_array = [] # used to be encoding info but commandeered for surrounding_recalls ~~~
    recall_position_array = []
    session_events = pd.DataFrame()
    
    trial_by_trial_correlation = []; elec_by_elec_correlation = []
    elec_ripple_rate_array = []
    channel_coords = []; electrode_labels = []

    semantic_clustering_key = []
    temporal_clustering_key = []
    semantic_clustering_from_key = []
    serialpos_lags = []; serialpos_from_lags = []; CRP_lags = []
    list_trial_nums = []; list_num_key = []
    list_level_semantic = []; list_level_temporal = []
    
    program_ran = 0
    
    try:        
#         with open(f'/scratch/john/SWRrefactored/patient_info/temp_dfSWR_{region_name}_{selected_period}_{exp}.p', 'rb') as f: ### change here to avoid overwrite
#             temp_df = dill.load(f)
            
#         row = None
#         # if temp_df_select is a patient id, then loop through the available patients
#         # until the selected patient id is found and retrieve the row in temp_df
#         # corresponidng to the selected patient
#         if isinstance(temp_df_select, str):
#             for r in temp_df:
#                 if temp_df_select == r.subject:
#                     row = r
#                     break
                    
#         elif isinstance(temp_df_select, int):
#             row = temp_df[temp_df_select]

        if row is None:
            print("temp df select failed to select a row")
            sys.exit()
        
        sub = row.subject; session = row.session; exp = row.experiment
        
        
        # add sessions to job_started file so that other slurm processes
        # won't run the same session
        print(f'{sub}-{session}')       
     
        mont = int(row.montage); loc = int(row.localization)
        reader = CMLReadDFRow(row)
        # events, provides description in a df format
        evs = reader.load('task_events')
    
        evs_free_recall = evs[(evs.type=='REC_WORD')] #& (evs.recalled==True)] # recalled word AND correct (from this list...no instrusions). 
                        # deal with intrusions after checking clustering to ensure removing them doesn't create fake transitions

        # need to remove free recalls that happened more than once
        word_evs = evs[evs['type']=='WORD'] # get words 
        

        session_words = word_evs.item_name.unique()
        if len(findAinB(['BARRO','CABALLO','COSTILLA','DULCE','FICHA','ESTUFA','GALLINA','FRIJOL',
                        'GUARDIA','HILO','JEFE','LABIO','MAIZ','NIEVE','ORO','PALMA','PECHO','PLATO',
                        'RANA','SAPO','SERPIENTE','TALLO','UVA','VIGA','ZAPATO'],session_words))>0:
            print("WORDS are in spanish")
            add_session_to_exclude_list("Words are in spanish")
            sys.exit() # skip Spanish sessions
            
        else:
            
            from sklearn.decomposition import PCA
            semantic_model = pickle.load(open('/home1/john/SWR/code/semantic_analysis/'+exp+'_wordpool_feats.pk', 'rb')) # dictionary of all exp words

            # get good lists to run through
            list_nums = evs_free_recall.list.unique()[evs_free_recall.list.unique()>0] # make sure not a practice list
            no_eeg_lists = evs_free_recall[evs_free_recall.eegoffset == -1].list.unique() # if any recalls in list don't have eeg remove that list
            list_nums = [i for i in list_nums if i not in no_eeg_lists] 
            
            list_mask = []
            temp_semantic_key = []
            temp_temporal_key = []
            temp_semantic_from_key = []
            temp_serialpos_lags = []; temp_from_lags = []; temp_CRP_lags = []
            temp_list_trial_nums = []; temp_list_num_key = []
            temp_list_level_semantic = []; temp_list_level_temporal = []
            
            for ln in list_nums:
                
                recalls_serial_pos = getSerialposOfRecalls(evs_free_recall,word_evs,ln) # note that intrusions become -999

                # get differences in lags before removing repeats
                list_serialpos_lags = np.append(np.diff(recalls_serial_pos),0) # absolute differences in lags with a 0 at end for last output
                list_from_lags = np.append(0,list_serialpos_lags[:-1]) # get lag from previous recall so I can look between recalls            
            
                # leave intrusions and repeats in for now so can assess clustering with them included
                # ...but still want to know how many legit recalls there are for min_recalls threshold
                temp_serialpos,_ = removeRepeatsBySerialpos(recalls_serial_pos) # remove repeats
                num_true_recalls = len(temp_serialpos[temp_serialpos!=-999])# how many non-intrusions (note that all but 1 -999s removed in last line)
                 
                if num_true_recalls >= min_recalls:
                    list_mask.append(ln) # update kept list mask
                    
                    # now get semantic and temporal values for this list
                    serial_pos = np.array(recalls_serial_pos)-1 # zero index the serialpos
#                   # serial_pos = [int(list_words_df[list_words_df['item_name']==w]['serialpos'])-1 for w in list_recalls_df['item_name']] # serialpos starting at 0
                    list_recalls_df = evs_free_recall[evs_free_recall.list==ln] # recalls df just for this list
                    list_words_df = word_evs[word_evs.list==ln] # words df just for this list
        
                    words = list(list_words_df['item_name']) 
                    if 'AXE' in words: # GoogleVec doesn't have this spelling of ax (fix for semantic clustering)
                        list_words_df = list_words_df.replace('AXE','AX')
                        list_recalls_df = list_recalls_df.replace('AXE','AX')
                    
                    ## semantic clustering ##
        
                    if exp == 'FR1':
                        # Decided not to use semantic for FR1 for now but if you do I think FR1_wordpool_feats.pk 
                        # needs to be recompiled via SWRgetWordList.ipynb 2020-10-19
                        # also should make sure semantic_transition_scores look kosher since made some 
                        # changes to get_recall_clustering to fix temporal clustering of intrusions
                
                        #Project semantic features for this list to 1 dimension
                        feats = np.array([semantic_model[w] 
                                          for w in list_words_df['item_name']
                                         ])  #construct feature matrix from this list; # WORDs X 300 vecs
                        pca = PCA(n_components=PCA_ndim)
                        pcs = pca.fit_transform(feats) # list of ndim PCs for 12 words

                        # Get semantic cluster transition pairwise values 
                        semantic_transition_scores = np.array(get_recall_clustering(pcs, serial_pos))
                        semantic_list_mean = np.nanmean(semantic_transition_scores[semantic_transition_scores > -990])
                        temp_semantic_key.extend(semantic_list_mean*np.ones(len(list_recalls_df))) # extend by # of recalls for key
                        
                        if len(list_recalls_df)>0:
                            temp_semantic_key[-1] = 'nan' # last recall in every list didn't lead to a transition. So don't give a value
                            
                    elif exp == 'catFR1':
                        # have to use special structure of catFR1 to divide into near v. remote semantic transitions
                        # there's also adjacent v. non-adjacent. so create 2x2 recalls as well as dead ends (no transition from last one)
                        # category numbers to identify semantic transitions
                        item_names = list_recalls_df['item_name']
                        category_nums = [] # don't use list comprehensions so can put -999s in for intrusions
                        for w in item_names:
                            if w in np.array(list_words_df['item_name']):
                                category_nums.append(int(list_words_df[list_words_df['item_name']==w]['category_num']))
                            else:
                                category_nums.append(-999)

                        # identity of 1st of words before transition
                        for recall in np.arange(len(serial_pos)):
                            # it's ok if intrusions or repeats are in here since will be removed later
                            if recall == len(serial_pos)-1: # if last one, no transition happens after this, so put a dummy letter
                                temp_semantic_key.extend('Z')
                            else:
                                transition_dist = np.abs(serial_pos[recall+1]-serial_pos[recall])
                                if transition_dist == 1:
                                    # use single letters and not words since I used extend for FR1 semantic values
                                    if category_nums[recall+1] == category_nums[recall]:
                                        temp_semantic_key.extend('A') # leads to adjacent semantic
                                    else:
                                        temp_semantic_key.extend('B') # leads to adjacent other
                                elif category_nums[recall+1] == -999:
                                    temp_semantic_key.extend('I') # this one leads to an intrusion next
                                elif item_names.iloc[recall] == item_names.iloc[recall+1]:
                                    temp_semantic_key.extend('R') # consecutive repeats
                                else: 
                                    if category_nums[recall+1] == category_nums[recall]:
                                        temp_semantic_key.extend('C') # leads to remote semantic
                                    else:
                                        temp_semantic_key.extend('D') # leads to remote other
                                        
                        # do the same for 2nd word of transition (to look between the clustered pair instead of before it)     
                        for recall in np.arange(len(serial_pos)):
                            if recall == 0: # if first one, didn't come from anywhere!
                                temp_semantic_from_key.extend('Z')
                            else:
                                transition_dist = np.abs(serial_pos[recall]-serial_pos[recall-1])
                                if transition_dist == 1:
                                    # use single letters and not words since I used extend for FR1 semantic values
                                    if category_nums[recall] == category_nums[recall-1]:
                                        temp_semantic_from_key.extend('A') # adjacent semantic transition
                                    else:
                                        temp_semantic_from_key.extend('B') # adjacent other
                                elif category_nums[recall] == -999:
                                    temp_semantic_from_key.extend('I') # intrusion transition
                                elif item_names.iloc[recall] == item_names.iloc[recall-1]:
                                    temp_semantic_from_key.extend('R') # consecutive repeats
                                else: 
                                    if category_nums[recall] == category_nums[recall-1]:
                                        temp_semantic_from_key.extend('C') # remote semantic transition
                                    else:
                                        temp_semantic_from_key.extend('D') # remote other    
                                        
                    ## do temporal clustering ##
                    
                    # sps are 0-indexed
                    temporal_transition_scores = np.array(get_recall_clustering(np.arange(len(list_words_df)), serial_pos))
                    # average those values that aren't intrusions/repeats 
                    temporal_list_mean = np.nanmean(temporal_transition_scores[temporal_transition_scores > -990])
                    temp_temporal_key.extend(temporal_list_mean*np.ones(len(list_recalls_df)))
                    if len(list_recalls_df)>0:
                        temp_temporal_key[-1] = 'nan' # last recall in every list didn't lead to a transition. So don't give a value
                        
                    # 2020-12-24 Mike thinks it's smart to remove lag=1 trials where they start with the first item
                    # i.e. 0->1, but if that exists, also check for 1->2, etc. to disqualify whole chain from 0 
                    if (len(serial_pos)>1) and (serial_pos[0] == 0): # already 0-indexed
                        current_pos = 1
                        while (serial_pos[current_pos] == serial_pos[current_pos-1]+1):
                            list_serialpos_lags[current_pos-1] = -995 # in case we want to select for these later give a unique value
                            list_from_lags[current_pos] = -995 # same idea for keeping track of lag from previous recall
                            current_pos+=1
                            if current_pos == len(serial_pos): # at end of list
                                break
                                
                    # 2020-12-30 we want to make a lag-CRP curve with -3:+3 and also bins for transitions to the first and 
                    # last items in the list. This is a different goal than just adjacent v. remote, so let's keep a separate
                    # vector of these special lags with -100 as transition to 1st item and -110 as transition to 12th item
                    list_CRP_lags = copy(list_serialpos_lags)
                    if (len(serial_pos)>1):
                        for i,sp in enumerate(serial_pos[1:]): # start from 2nd serial_pos which will fill back the index 1-back
                            if sp == 0: # already --indexed
                                list_CRP_lags[i] = -100
                            elif sp == 11:
                                list_CRP_lags[i] = -110
                    temp_CRP_lags.extend(list_CRP_lags)

                    # keep track of lags
                    temp_serialpos_lags.extend(list_serialpos_lags)
                    temp_from_lags.extend(list_from_lags)
                    
                    # keep track of number of trials in each so can select by minimum later on
                    temp_list_trial_nums.append(len(list_recalls_df))
                    temp_list_num_key.append(ln)
                    
                    # for whole_retrieval analysis need list-level scores for temporal/semantic clustering
                    temp_list_level_temporal.append(temporal_list_mean) # single value for each list
                    if exp == 'FR1':
                        temp_list_level_semantic.append(semantic_list_mean) # they're list_level values for FR1 so just grab first of list
                    else: # for catFR keep list of lists since will have to go through later to assess minimum As or Cs or whatever
                        temp_list_level_semantic.append(temp_semantic_key[-len(serial_pos):]) # grab all of them for this list since will weigh A v B v C v D later
            
        evs_free_recall = evs_free_recall[evs_free_recall.list.isin(list_mask)] # get recalls from qualifying lists
        # need to remove free recalls that happened more than once (list_mask doesn't account for these)        
#         evs_free_recall = removeRepeatedRecalls(evs_free_recall,word_evs)
#         evs_free_recall = evs_free_recall[evs_free_recall.eegoffset > -1]
      
        # select which recalls??         
        [recall_selection_name,selected_recalls_idxs] = selectRecallType(recall_type_switch,evs_free_recall,IRI,recall_minimum)
        
        # now have recalls you want after including intrusions and repeats for clustering & IRI purposes
        # ...so remove them and update selected_recall_idxs (since this is key to select clustering values down below)
        _,nonrepeat_indicator = removeRepeatedRecalls(evs_free_recall,word_evs)
        nonintrusion_idxs = np.array(evs_free_recall.intrusion==0)
        
        selected_recalls_idxs = selected_recalls_idxs & (nonrepeat_indicator>0) & nonintrusion_idxs

        evs_free_recall = evs_free_recall[selected_recalls_idxs]

        if len(evs_free_recall)==0:
            print("Length of evs free recall is 0")
            add_session_to_exclude_list("Length of evs free recall is 0")
            sys.exit()
        
        # get localizations (region info)
        pairs = reader.load('pairs')
        try:
            localizations = reader.load('localization')
        except:
            localizations = []
        tal_struct, bipolar_pairs, mpchans = get_bp_tal_struct(sub, montage=mont, localization=loc)
        elec_regions,atlas_type,pair_number,has_stein_das = get_elec_regions(localizations,pairs) 
        
        # load eeg
        if selected_period == 'surrounding_recall':
            total_recalls = total_recalls + len(evs_free_recall) # get total recalls from lists
            total_lists = total_lists + len(evs[evs.type=='WORD'].list.unique()) # get total lists
            kept_recalls = kept_recalls + len(evs_free_recall)
            eeg_events = evs_free_recall
            
            # fix EEG offset due to Unity implementation error
    #         init_time = eeg_events.iloc[0].eegoffset
            eeg_events = correctEEGoffset(sub,session,exp,reader,eeg_events)
    #         print(sub+'-'+str(session)+'-'+exp+': '+str(eeg_events.iloc[0].eegoffset-init_time)) 
    
        elif selected_period == 'whole_retrieval':
            # grab whole retrieval periods for a better baseline of SWRs
            evs_rets = evs[evs.type=='REC_START']
#             evs_rets = evs_rets[evs_rets.eegoffset>-1] # any trial with no eeg gets removed by cmlreaders so it's not in ripple_array 
#             evs_rets = evs_rets[evs_rets.list>-1] # remove practice lists
            # in clustering I make a list_mask, so use that to select which trials to get EEG for
            evs_rets = evs_rets[evs_rets.list.isin(list_mask)]
            eeg_events = evs_rets
            
            # get alignmnet of end of beep time to EEG so can align retrieval to end of beep across all sessions
            align_adjust = getRetrievalStartAlignmentCorrection(sub,session,exp) # in ms
            
        elif selected_period == 'encoding':
            # I'm going to save encoding word events too, but need a mask to keep track of:
            # 0) words not recalled 1) words recalled from this list 2) words later recalled BUT IRI<2 s so removed
            evs_encoding_words = evs[evs.type=='WORD']
            evs_encoding_words = evs_encoding_words[evs_encoding_words.list>-1]            
            evs_encoding_words = evs_encoding_words[evs_encoding_words.eegoffset>-1]
            encoded_word_key = np.zeros(len(evs_encoding_words)) # 0 for not recalled
            encoded_word_key[evs_encoding_words.recalled==True] = 2 # 2 for recalled but removed bc IRI<2 s
            encoded_word_key[evs_encoding_words.item_name.isin(evs_free_recall.item_name.unique())] = 1 # recalled words
            # since finding all encoding words IN the list of correctly free recalled words won't have any intrusions
            eeg_events = evs_encoding_words
            
            # for encoding analyses I want to know the semantic or temporal label of each recall
            orig_evs_free_recall = evs[(evs.type=='REC_WORD') & (evs.list>-1) & (evs.eegoffset != -1)] # original evs_free_recall is same len as temp_semantic_key
            if exp == 'catFR1':
                enc_semantic_key = []                          
                for ln in np.unique(eeg_events.list.values):
                    num_words = sum(eeg_events.list==ln)
                    recall_list_idxs = findInd(orig_evs_free_recall.list==ln)
                    if len(recall_list_idxs)>0:
                        list_semantic_key = np.array(temp_semantic_key)[recall_list_idxs]
                    else:
                        list_semantic_key = [] # if no recalls just save repeated empty matrices to keep idxs aligned
                    enc_semantic_key += num_words * [list_semantic_key] # To add v, n times, to l: l += n * [v]
            elif exp == 'FR1':
                enc_temporal_key = []                          
                for ln in np.unique(eeg_events.list.values):
                    num_words = sum(eeg_events.list==ln)
                    recall_list_idxs = findInd(orig_evs_free_recall.list==ln)
                    if len(recall_list_idxs)>0:
                        list_temporal_key = np.array(temp_serialpos_lags)[recall_list_idxs]
                    else:
                        list_temporal_key = [] # if no recalls just save repeated empty matrices to keep idxs aligned
                    enc_temporal_key += num_words * [list_temporal_key] # To add v, n times, to l: l += n * [v]            
            
        # fixing bad trials
        if sub == 'R1045E' and exp == 'FR1': # this one session has issues in eeg trials past these points so remove events
            if selected_period == 'surrounding_recall':
                sys.exit() # too difficult to deal with for clustering analysis   
                # eeg_events = eeg_events.iloc[:65,:] # only the first 66 recalls have good eeg
            elif selected_period == 'whole_retrieval':
                eeg_events = eeg_events.iloc[:20,:] # only the first 20 retrieval periods have good eeg
            elif selected_period == 'encoding':
                eeg_events = eeg_events.iloc[:263,:] # same idea
                encoded_word_key = encoded_word_key[:263]
                
        # note I added the align_adjust now for whole_retrieval where I adjust all retrieval starts to beep_end
        eeg = reader.load_eeg(events=eeg_events, rel_start=psth_start-eeg_buffer+align_adjust, 
                              rel_stop=psth_end+eeg_buffer+align_adjust, clean=True, scheme=pairs) 
        
        
        # events X electrodes X time
        sr = eeg.samplerate

        # if weird samplerate, add a few ms to make the load work
        if (499<sr<500) | (998<sr<1000):
            time_add = 1
            if (499<sr<500):
                sr = 500
            elif (998<sr<1000):
                sr = 1000
            while eeg.shape[2] < (psth_end-psth_start+2*eeg_buffer)/(1000/sr):
                eeg = reader.load_eeg(events=eeg_events, rel_start=int(psth_start-eeg_buffer+align_adjust), 
                                      rel_stop=int(psth_end+eeg_buffer+time_add+align_adjust), clean=True, scheme=pairs)
                if time_add>50: #** 
                    print("Time add is greater than 50")
                    add_session_to_exclude_list("Time add is greater than 50")
                    sys.exit()
                time_add+=1
            time_add_save.append(time_add)
            eeg.samplerate = sr # need to overwrite those that were just fixed
            
        eeg_mne = eeg.to_mne()
        eeg = None # clear variable
        
        # only analyze data in the region of interest
        selected_elec = []
        selected_elec_regions = []
        for idx, elec_region in enumerate(elec_regions):
            if elec_region in selected_region:
                selected_elec.append(idx)
                selected_elec_regions.append(elec_region)
                
        # get bad channel mask
        try:
            elec_cats = reader.load('electrode_categories') # this is cool
        except:
            if remove_soz_ictal == True:
                print("Remove soz ictal is true")
                add_session_to_exclude_list("Remove soz ictal is true")
                sys.exit() # don't know soz/ictal sites so skip this session
            else: 
                elec_cats = [] # not removing these sites anyway so keep on keeping on
#             e = 'No electrode categories for '+sub+', session '+str(session)
#             LogDFExceptionLine(row, e, 'SWR_get_eeg_log.txt')

        bad_bp_mask = getBadChannels(tal_struct,elec_cats,remove_soz_ictal)
        bad_electrode_idxs = np.argwhere(bad_bp_mask!=0)
        
        # remove bad electrodes 
        selected_elec = np.setdiff1d(selected_elec, bad_electrode_idxs)
        eeg_mne.pick(selected_elec)
        
        # downsample sr to 500 
        if sr > 500:
            sr = 500
            eeg_mne = eeg_mne.resample(sfreq=sr)
            
        elif sr == 500:
            pass
        
        else:
            print("Sampling rate is too low: ", sr)
            add_session_to_exclude_list("Sampling rate is too low")
            sys.exit()
        
        #eeg_mne = eeg_mne.filter(l_freq=62, h_freq=58, method='iir', iir_params=dict(ftype='butter', order=4), n_jobs=n_jobs)
        
        if testing_mode:
            eeg_mne.save('tutorials/eeg_mne_test-epo.fif')
        
        ## FILTERS ##
        trans_width = 5.
        ntaps = (2/3)*np.log10(1/(10*(1e-3*1e-4)))*(sr/trans_width) # gives 400 with sr=500, trans=5
        if sr == 512 or sr == 1024 or sr == 1023.999: # last one fixes R1221P
            ntaps = np.ceil(ntaps)
        FIR_bandstop = firwin(int(ntaps+1), [70.,178.], fs=sr, window='hamming',pass_zero='bandstop')
        bandstop_25_60 = firwin(int(ntaps+1), [20.,58.], fs=sr, window='hamming',pass_zero='bandstop') # Norman 2019 IED
        nyquist = sr/2
        ntaps40, beta40 = kaiserord(40, trans_width/nyquist)
        kaiser_40lp_filter = firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, nyq=nyquist, pass_zero='lowpass')

        eeg_rip_band = eeg_mne.copy()
        eeg_ied_band = eeg_mne.copy()
        eeg_rip_band._data = eeg_rip_band._data-filtfilt(FIR_bandstop,1.,eeg_rip_band._data)
        eeg_ied_band._data = eeg_ied_band._data-filtfilt(bandstop_25_60,1.,eeg_ied_band._data)
        
        eeg_rip_band.apply_hilbert(envelope=True)
        eeg_ied_band.apply_hilbert(envelope=True)
        
        # go through each channel in given region and detect ripples!
        region_electrode_ct = 0
        selected_elec_ripples_idxs = []
        for channel in range(len(selected_elec)):
                
            print('Using channel '+str(channel)+' from '+sub+', '+str(session))

            # get data from MNE container
            eeg_rip = eeg_rip_band.get_data()[:,channel,:]     
            eeg_ied = eeg_ied_band.get_data()[:,channel,:]

            # select detection algorithm (note that iedlogic is same for both so always run that)
            eeg_ied = eeg_ied**2 # already rectified now square
            eeg_ied = filtfilt(kaiser_40lp_filter,1.,eeg_ied) # low pass filter  
            mean1 = np.mean(eeg_ied)
            std1 = np.std(eeg_ied)
            iedlogic = eeg_ied>=mean1+4*std1 # Norman et al 2018     
                    
            # detect ripples
            ripplelogic = detectRipplesHamming(eeg_rip,trans_width,sr,iedlogic)   
            if sr>desired_sample_rate: # downsampling here for anything greater than 500
                ripplelogic = downsampleBinary(ripplelogic,sr/desired_sample_rate)
                
            # ripples are detected, so can remove buffers now #**
            ripplelogic = ripplelogic[:,int(eeg_buffer/(1000/desired_sample_rate)):int((psth_end-psth_start+eeg_buffer)/(1000/desired_sample_rate))]  

            # skip this electrode if the ripple rate is below threshold
            temp_start_array,_ = getStartEndArrays(ripplelogic)
            elec_ripple_rate = np.sum(temp_start_array)/temp_start_array.shape[0]/((psth_end-psth_start)/1000)
            if elec_ripple_rate < min_ripple_rate:
                print(sub+', '+str(session)+' skipped b/c below ripple rate thresh for ch.: '+str(channel))
                #continue
            elif elec_ripple_rate > max_ripple_rate:
                print(sub+', '+str(session)+' skipped b/c ABOVE ripple rate thresh for ch.: '+str(channel))
                # skip this electrode

            # check the ripples for this electrode and make sure they're not super correlated across trials
            # first, bin the array so can get more realistic correlation not dependent on ms timing
            binned_ripplelogic = downsampleBinary(ripplelogic,10) # downsample by 10x so 10 ms bins
            trial_ripple_df = pd.DataFrame(data=np.transpose(binned_ripplelogic))
            num_cols = len(list(trial_ripple_df))
            trial_ripple_df.columns = ['col_' + str(i) for i in range(num_cols)] # generate range of ints for suffixes
            temp_tbt_corr = np.mean(pg.pairwise_corr(trial_ripple_df,method='spearman').r)
            
            # save these even if removing
            trial_by_trial_correlation.append(temp_tbt_corr) # corr b/w trials
            
            if temp_tbt_corr > max_trial_by_trial_correlation:
                print(sub+', '+str(session)+' skipped b/c above trial-by-trial correlation for ch.: '+str(channel))
                #continue
            elec_ripple_rate_array.append(elec_ripple_rate) # record the average ripple rate for this electrode 
    
            ## if this electrode passes SAVE data ##
            
#                 if selected_period == 'surrounding_recall': 
#                     if len(np.array(temp_semantic_key)[selected_recalls_idxs]) != len(ripplelogic):
#                         # eegoffsets with -1 should be removed above anyway but keep as a check
#                         sys.exit()
    
            # append arrays across electrodes
            ripple_array.append(ripplelogic)
            selected_elec_ripples_idxs.append(channel)
            session_ripple_rate_by_elec = superVstack(session_ripple_rate_by_elec,np.mean(binned_ripplelogic,0)) # for correlation b/w elecs below
            
            # get other info specific to task periods         
            if selected_period == 'encoding':
                encoded_word_key_array.append(encoded_word_key) # save the key for each electrode so easier to unpack later
                serialpos_array.append(eeg_events.serialpos)
                if len(encoded_word_key) != ripplelogic.shape[0]:
                    e = 'Encoded word key and ripple_array dont match for '+sub+str(session)+'_'+str(channel)
                    LogDFExceptionLine(row, e, 'SWR_get_eeg_log.txt')
                if exp == 'catFR1':
                    category_array.extend(np.array(eeg_events.category))
                list_num_key.extend(eeg_events.list)                    
                session_events = session_events.append(eeg_events)
                
                # for every word place the recall output positions
                temp_evs_free_recall = evs[(evs.type=='REC_WORD')&(evs.list>-1)]
                list_nums = eeg_events.list.unique() 
                for ln in list_nums:
                    num_words = sum(eeg_events.list==ln)
                    recall_position_array += num_words * [getSerialposOfRecalls(temp_evs_free_recall,eeg_events,ln)] # To add v, n times, to l:
                
                # repurpose this value from recall to know identity of recalls for every word in scope of encoding
                if exp == 'catFR1':
                    semantic_clustering_key+=enc_semantic_key
                elif exp == 'FR1':
                    temporal_clustering_key+=enc_temporal_key
                    
            if selected_period == 'whole_retrieval':
                # make a matrix of the times of free recall if looking at whole retrieval period
                list_nums = evs_rets.list.unique() # count all list numbers for this session
                temp_fr_array = np.zeros((len(list_nums),retrieval_whole_time),dtype=np.int8)
                for ln in list_nums:
                    list_times = np.array(evs_free_recall[evs_free_recall.list==ln].mstime) - \
                                (np.array(evs_rets[evs_rets.list==ln].mstime) + align_adjust) # align all times to end of beep
#                         list_times = evs_free_recall[evs_free_recall.list==ln].rectime # I confirmed these are accurate relative to REC_START
                    for i,list_time in enumerate(list_times):
                        if list_time <= retrieval_whole_time:
                            # used to do ln-1 instead of i to get order opposite but doesn't work now since I remove lists
                            temp_fr_array[i][int(np.round(list_time))] = 1 
                fr_array = superVstack(fr_array,temp_fr_array)
                list_num_key.extend(temp_list_num_key)
                
            elif selected_period == 'surrounding_recall':    # ~~~                                
                # adding new values 2020-11-24 for some suggested analyses from group

                # key of serialpos for recalls
                list_nums = evs_free_recall.list.unique()   
                temp_recalls_serialpos = []
                for ln in list_nums:
                    temp_sp = getSerialposOfRecalls(evs_free_recall,word_evs,ln)
                    temp_recalls_serialpos.extend(temp_sp)
                    # recall number per list
                    list_recall_num_array.extend(np.tile(len(temp_sp),len(temp_sp)))
                    # recall position per list
                    recall_position_array.extend(np.arange(len(temp_sp))+1) # 1-indexed
                    # list_nums
                    list_num_key.extend(np.tile(ln,len(temp_sp)))
                serialpos_array.extend(np.array(temp_recalls_serialpos))

                # key of rectimes for recalls
                rectime_array.extend(np.array(evs_free_recall.rectime))  
                
                # just save the whole dataframe too so I have name and list_num (so I can align to Jim/David's analysis 2021-10-05)
                session_events = session_events.append(eeg_events)    
                
                # put these here since for encoding now going to use semantic_clustering_key at that scope
                semantic_clustering_key.extend(np.array(temp_semantic_key)[selected_recalls_idxs])
                temporal_clustering_key.extend(np.array(temp_temporal_key)[selected_recalls_idxs])  
                if exp == 'catFR1':
                    semantic_clustering_from_key.extend(np.array(temp_semantic_from_key)[selected_recalls_idxs])                    

            serialpos_lags.extend(np.array(temp_serialpos_lags)[selected_recalls_idxs])
            serialpos_from_lags.extend(np.array(temp_from_lags)[selected_recalls_idxs])
            CRP_lags.extend(np.array(temp_CRP_lags)[selected_recalls_idxs])

            # record list-level numbers (for whole_retrieval analysis)
            list_trial_nums.extend(temp_list_trial_nums)                 
            list_level_temporal.extend(temp_list_level_temporal)
            list_level_semantic.extend(temp_list_level_semantic)
            
            trial_nums = np.append(trial_nums,ripplelogic.shape[0])
            region_electrode_ct+=1 # channel ct for this session
            elec_names.append(selected_elec_regions[channel])
            sub_sess_names.append(sub+'-'+str(session))
            sub_names.append(sub)
            electrode_labels.append(pairs.iloc[channel].label) # get names of electrodes so can look for them across sessions
            
            # get atlas coordinates for this electrode
            if 'avg.x' in pairs:
                temp_coord = np.append(pairs.iloc[channel]['avg.x'],np.append(pairs.iloc[channel]['avg.y'],pairs.iloc[channel]['avg.z']))
            elif 'ind.x' in pairs:
                temp_coord = np.append(pairs.iloc[channel]['ind.x'],np.append(pairs.iloc[channel]['ind.y'],pairs.iloc[channel]['ind.z']))
            else:
                temp_coord = np.empty(3); temp_coord[:] = np.nan
            channel_coords.append(temp_coord)              

        program_ran = 1  
                
        # before we keep all the electrodes in this session, remove electrodes that are highly correlated.
        # this really only makes sense for the cluster version, since I only do a session at a time here
#         if np.sum(session_ripple_rate_by_elec)>0 and region_electrode_ct>1:
#             session_ripple_df = pd.DataFrame(data=np.transpose(session_ripple_rate_by_elec))
#             num_cols = len(list(session_ripple_df))
#             session_ripple_df.columns = ['col_' + str(i) for i in range(num_cols)] # generate range of ints for suffixes
#             elec_by_elec_correlation = np.mean(pg.pairwise_corr(session_ripple_df,method='spearman').r) # correlation b/w elecs 
            #if elec_by_elec_correlation > max_electrode_by_electrode_correlation:
            #    program_ran = 0
            #    add_session_to_exclude_list("Program ran 0")
            #    sys.exit()
                            
        #if len(selected_elec_ripples_idxs) == 0:
        #    add_session_to_exclude_list("Ripple array is empty")
        #    sys.exit()

    except Exception as e:
        add_session_to_exclude_list(f"An exception occurred: {e}")
        LogDFExceptionLine(row, e, 'ClusterRunSWR_log.txt') 
          
        
    if save_values == 1 and program_ran == 1:
    
        # get strings for path name for save and loading cluster data
        try:
            soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,recall_type_switch,selected_period,recall_minimum)
            path_name = save_path+subfolder
            
            if os.path.isdir(path_name) == False:
                os.makedirs(path_name)
                
            if testing_mode:
                fn = os.path.join(path_name, 'test.p')
            else:
                fn = os.path.join(path_name,
                'SWR_'+exp+'_'+sub+'_'+str(session)+'_'+region_name+'_'+selected_period+recall_selection_name+
                            '_'+soz_label+'_'+filter_type+'.p')  # str(min_recalls)+'.p') # don't need this now take care of min after loading # '-ZERO_IRI.p'
            
            print("SAVE FILE NAME: ", fn)    
            
            with open(fn,'wb') as f:
                pickle.dump({'region_electrode_ct':region_electrode_ct, 
                            'elec_names':elec_names, 'sub_sess_names':sub_sess_names,
                            'raw_eeg':eeg_mne._data,
                            'ripple_array':np.stack(ripple_array, axis=1),
                            'time_add_save':time_add_save,
                            'trial_nums':trial_nums, 
                            'encoded_word_key_array':encoded_word_key_array,'category_array':category_array,
                            'serialpos_array':serialpos_array,'list_recall_num_array':list_recall_num_array, # ~~
                            'rectime_array':rectime_array,'session_events':session_events,
                            'recall_position_array':recall_position_array, 
                            'fr_array':fr_array, 'sub_names':sub_names,
                            'total_recalls':total_recalls, 'kept_recalls':kept_recalls,
                            'trial_by_trial_correlation':trial_by_trial_correlation, # one value for each electrode for this session
                            'elec_by_elec_correlation':elec_by_elec_correlation,
                            'elec_ripple_rate_array':elec_ripple_rate_array, #'recall_before_intrusion_array':recall_before_intrusion_array,
                            'semantic_clustering_key':semantic_clustering_key,'temporal_clustering_key':temporal_clustering_key,
                            'semantic_clustering_from_key':semantic_clustering_from_key,
                            'serialpos_lags':serialpos_lags,'serialpos_from_lags':serialpos_from_lags,'CRP_lags':CRP_lags,
                            'list_trial_nums':list_trial_nums,'list_num_key':list_num_key,
                            'list_level_semantic':list_level_semantic, 'list_level_temporal':list_level_temporal,
                            'electrode_labels':electrode_labels,'channel_coords':channel_coords}, f) # no channel_nums for wahtever reason
            print("Saved data")
            
        except:
            LogDFExceptionLine(row, e, 'ClusterRunSWR_log.txt') 
            add_session_to_exclude_list("Could not save file")
            
    else:
        print("Program ran: ", program_ran)
        print("Save values: ", save_values)  
        
        
##############################################################################
### load sub_df made in getSubDFforSlurm.ipynb and run every session in it ###    
##############################################################################
                               
                                
cluster_run = 1 # run all patients in parallel. If ==0, define subject/region_list/session below
replace_files = 0 # replace the files that exist already on scratch

    
with open(f'/scratch/john/SWRrefactored/patient_info/temp_dfSWR_{selected_period}_{exp}.p', 'rb') as f:
    temp_df = dill.load(f)

if cluster_run == 0:
    subject = 'R1367D' #'R1385E'
    region_list = HPC_labels
    session = 0
    
    for row in temp_df:
        if ((row.subject==subject) & (row.session==session)):
            break
        else:
            row = []
    if row == []:
        print(f'Session {session} does not exist for {subject} in this df')
    else:
        ClusterRunSWRs(row,region_list,save_path=save_path,
                   selected_period=selected_period,exp=exp)
else:
    task_id = int(sys.argv[1])    
    if task_id >= len(temp_df):
        print(f"Task ID {task_id} is out of range. Maximum allowed index is {len(temp_df) - 1}.")
        sys.exit(1)
    row = temp_df[task_id] # slurm 

    # e.g. region_str is string "HPC_labels" and region_list is imported list HPC_labels 
    for region_str, region_list in available_regions.items(): 
        region_shorthand = region_str.replace('_labels', '')

        print(f"Running script for patient {row.subject}, session {row.session} region {region_shorthand}, and exp {exp}")

        # Generate file name
        soz_label, recall_selection_name, subfolder = getSWRpathInfo(remove_soz_ictal, recall_type_switch, selected_period, recall_minimum)
        path_name = save_path + subfolder
        fn = os.path.join(path_name,
                          'SWR_' + exp + '_' + row.subject + '_' + str(row.session) + '_' + region_shorthand + '_' +
                          selected_period + recall_selection_name + '_' + soz_label + '_' + filter_type + '.p')

        if os.path.exists(fn):
            if replace_files == 1:
                print(f"File {fn} exists and will be overwritten.")
            else:
                print(f"File {fn} exists and replace_files is not set to 1. Exiting.")
                sys.exit(1)          


        ClusterRunSWRs(row,region_list,save_path=save_path,
                       selected_period=selected_period,exp=exp)

        print(f"DONE for patient {row.subject}, session {row.session} region {region_shorthand}, and exp {exp}")
