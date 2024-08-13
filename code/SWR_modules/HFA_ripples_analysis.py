
import pandas as pd; pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
import numpy as np
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import ButterworthFilter, ResampleFilter, MorletWaveletFilter
import xarray as xarray
import sys
import os
import matplotlib.pyplot as plt
from pylab import *
from scipy.stats import zscore
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
sys.path.append('/home1/john/johnModules')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                         MFG_labels, IFG_labels, nonHPC_MTL_labels
from general import *
from SWRmodule import *
import statsmodels.formula.api as smf
from ripples_HFA_general import HFA_ripples_prepare_data
from sklearn.metrics import r2_score


class HFA_ripples_analysis(HFA_ripples_prepare_data):
    
    def __init__(self, exp, df, sub_selection, select_subfield, hpc_regions,
                 ripple_bin_start_end=[100,1700], HFA_bins=[400,1100], ripple_regions=['ca1']):

        super().__init__(exp=exp, df=df, sub_selection=sub_selection,
                        select_subfield=select_subfield, hpc_regions=hpc_regions, 
                        ripple_bin_start_end=ripple_bin_start_end, 
                        HFA_bins=HFA_bins, ripple_regions=ripple_regions)
        
        self.psth_start = self.pre_encoding_time
        
    def remove_subject_sessions(self):
        
        super().remove_subject_sessions()
    
    def load_data_from_cluster(self, base_path, selected_period, ripple_bool, region_name, hpc_ripple_type):
        
        super().load_data_from_cluster(base_path, selected_period, ripple_bool = ripple_bool, 
                                       hpc_ripple_type = hpc_ripple_type, region_name = region_name)
        
    def getMixedEffectMeanSEsStartArray(self):
        
        super().getStartArray()
        
    def create_semantic_clustered_array(self, clustered=['A','C'], unclustered=['D', 'Z']):
        
        '''
        :param list clustered: indicates what recalls count as clustered. There are four possible recalls:
            1) 'A': adjacent sentic
            2) 'C': remote semantic
            3) 'D': remote unclustered
            4) 'Z': dead end 
        The default is to use A and C as clustered, and D and Z as unclustered. 
        '''
        
        self.number_of_lists = int(self.num_selected_trials/self.list_length)
        self.semantic_clustered_array_np = np.zeros((self.number_of_lists, self.list_length))
        self.dead_ends = 0
        self.remote_semantic = 0
        self.adjacent_semantic = 0
        self.remote_nonsemantic = 0
        self.adjacent_nonsemantic = 0
        counter = 0
        
        for list_idx in range(0, self.num_selected_trials, self.list_length):
            
            # recall_position_np and semantic_array_np contain information 
            # about the word that was recalled and its clustering type, respectively
            # this information is repeated list_length times, so our for loop will 
            # increment by list length 
            recalled_idx = self.recall_position_np[list_idx] 
            cluster = self.semantic_array_np[list_idx]
            
            # init values to -1 so that non recalled items are -1 
            cluster_trial_np = np.ones(self.list_length)*-1
            
            for r, c in zip(recalled_idx, cluster):
                if r > 0 and r <= self.list_length:
                    if c in clustered:
                        cluster_trial_np[r-1] = 1 # change to 1 for clustered recall
                    elif c in unclustered:
                        cluster_trial_np[r-1] = 0 # change to 0 for unclustered but recalled
                    if c=='A':
                        self.adjacent_semantic += 1
                    if c=='C':
                        self.remote_semantic += 1
                    if c=='D':
                        self.remote_nonsemantic += 1
                    if c=='Z':
                        self.dead_ends += 1
                    
            self.semantic_clustered_array_np[counter] = cluster_trial_np
            
            counter += 1
            
        self.semantic_clustered_array_np = np.ravel(self.semantic_clustered_array_np)
        
    def create_temporal_clustered_array(self):
        
        self.number_of_lists = int(self.num_selected_trials/self.list_length)
        self.temporal_clustered_array_np = np.zeros(self.num_selected_trials)
        self.temporal_clustered_array_np[:] = np.nan
        self.temporal_clustered_binary_array_np = np.zeros(self.num_selected_trials)
        self.temporal_clustered_binary_array_np[:] = np.nan
        self.temporal_clustered_binary_hp_array_np = np.zeros(self.num_selected_trials)
        self.temporal_clustered_binary_hp_array_np[:] = np.nan
                
        for list_idx in range(0, self.num_selected_trials, self.list_length):
            
            recalled_idx = self.recall_position_np[list_idx] 
            cluster = self.temporal_array_np[list_idx]
            recalled_idx_correct = [r for r in recalled_idx if r > 0 if r < 13]
            num_correct = len(recalled_idx_correct)

            # init values to np.nan
            cluster_trial_np = np.ones(self.list_length)
            cluster_trial_np[:] = np.nan
            
            for r, c in zip(recalled_idx, cluster):
                if r > 0 and r <= self.list_length: 
                    if np.abs(c) > 0 and np.abs(c) <= self.list_length:
                        idx = list_idx + (r-1)
                        self.temporal_clustered_array_np[idx] = c
                        if np.abs(c) == 1:
                            self.temporal_clustered_binary_array_np[idx] = 1
                            if num_correct > 6:
                                self.temporal_clustered_binary_hp_array_np[idx] = 1
                        if np.abs(c) > 3:
                            self.temporal_clustered_binary_array_np[idx] = 0
                            if num_correct > 6:
                                self.temporal_clustered_binary_hp_array_np[idx] = 0
                        
    def plot_SCE(self, freq_band, mode, title_str, savePath):
        
        # text for plot label
        cat_zero_text = 'Clustered'
        cat_one_text = 'Unclustered'
          
        if freq_band == 'H':
            neural_data = self.HFA_array_np
            ylabel = 'HFA activity'
        if freq_band == 'T':
            neural_data = self.theta_array_np
            ylabel = 'Theta activity'
        if freq_band == 'R':
            neural_data = self.ripple_freq_array_np
            ylabel = 'Ripple band activity'
        if freq_band == 'G':
            neural_data = self.gamma_array_np
            ylabel = 'Gamma activity'
            
        if mode == 'T':
            partition_array = self.temporal_clustered_binary_array_np
        if mode == 'THP':
            partition_array = self.temporal_clustered_binary_hp_array_np
        if mode == 'S':
            partition_array = self.semantic_clustered_array_np
            
        plot_ME_mean = 1 # 0 for typical PSTH, 1 for ME mean, 2 for average across sub averages
        
        # set up the PVTH stats parameters here too (for encoding have 30 bins)
        psth_start = int(self.pre_encoding_time/self.bin_size)
        psth_end = int(self.encoding_time/self.bin_size)

        bin_centers = np.arange(psth_start+0.5,psth_end)

        start_array_enc_0 = neural_data[partition_array==0] # not recalled/ unclustered (depending on analysis_type)
        start_array_enc_1 = neural_data[partition_array==1] # recalled/clustered (depending on analysis_type)

        sub_0 = self.subject_name_array_np[partition_array==0]
        sess_0 = self.session_name_array_np[partition_array==0]
        sub_1 = self.subject_name_array_np[partition_array==1]
        sess_1 = self.session_name_array_np[partition_array==1]

        PSTH_all = triangleSmooth(np.mean(neural_data,0),self.smoothing_triangle)
        min_val_PSTH = np.min(PSTH_all)
        max_val_PSTH = np.max(PSTH_all)
        
        self.generate_SCE_SME_plots(start_array_enc_1, start_array_enc_0, sub_1, sess_1, 
                                    sub_0, sess_0, cat_one_text, cat_zero_text, ylabel, plot_ME_mean, bin_centers,
                                    title_str, min_val_PSTH, max_val_PSTH, savePath)
        
            
    def generate_SCE_SME_plots(self, start_array_enc_1, start_array_enc_0, sub_1, sess_1, sub_0, sess_0, 
                               cat_one_text, cat_zero_text, ylabel_str, plot_ME_mean, bin_centers, title_str, 
                               min_val_PSTH, max_val_PSTH, savePath):
        
        pad = int(np.floor(self.smoothing_triangle/2)) 
        
        # loop through recalled and then forgotten words
        for category in range(2):
            
            if category == 0:
                temp_start_array = start_array_enc_1
                sub_name_array = sub_1
                sess_name_array = sess_1

                # HFA 
                PSTH = triangleSmooth(np.mean(temp_start_array,0),self.smoothing_triangle)
                subplots(1,1,figsize=(5,4))
                plot_color = (0,0,1)
                num_words = f"{cat_zero_text}: {temp_start_array.shape[0]}"

            else:       
                temp_start_array = start_array_enc_0
                sub_name_array = sub_0
                sess_name_array = sess_0
                
                PSTH = triangleSmooth(np.mean(temp_start_array,0),self.smoothing_triangle)

                plot_color = (0,0,0)
                num_words = f"{cat_one_text}: {temp_start_array.shape[0]}"

            # note that output is the net ± distance from mean
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")    
                mean_plot,SE_plot = getMixedEffectMeanSEs(temp_start_array,sub_name_array,sess_name_array)

            if plot_ME_mean == 1:
                PSTH = triangleSmooth(mean_plot,self.smoothing_triangle) # replace PSTH with means from ME model (after smoothing as usual)  
            elif plot_ME_mean == 2:
                temp_means = []
                for sub in np.unique(sub_name_array):
                    temp_means = superVstack(temp_means,np.mean(temp_start_array[np.array(sub_name_array)==sub],0))
                PSTH = triangleSmooth(np.mean(temp_means,0),self.smoothing_triangle)   
            
            xr = bin_centers #np.arange(psth_start,psth_end,binsize)
            if pad > 0:
                xr = xr[pad:-pad]
                binned_start_array = temp_start_array[:,pad:-pad] # remove edge bins    
                PSTH = PSTH[pad:-pad]
                SE_plot = SE_plot[:,pad:-pad]        
            
            plot(xr,PSTH,color=plot_color, label=num_words)
            fill_between(xr, PSTH-SE_plot[0,:], PSTH+SE_plot[0,:], alpha = 0.3)
            xticks(np.arange(self.pre_encoding_time+pad*100,self.encoding_time-pad*100+1,500)/100,
                np.arange((self.pre_encoding_time+pad*100)/1000,(self.encoding_time-pad*100)/1000+1,500/1000))
            xlabel('Time from word presentation (s)')
            ylabel(f'{ylabel_str} activity (z-scored)')
            title(title_str)
            tight_layout()
            ax = plt.gca()
            if min_val_PSTH < -1.0:
                lower_val = math.floor(min_val_PSTH)
            else:
                lower_val = -1.0
            if max_val_PSTH > 1.0:
                upper_val = math.ceil(max_val_PSTH)
            else:
                upper_val = 1.0
            ax.set_ylim(lower_val, upper_val)
            ax.set_xlim(self.pre_encoding_time/100,self.encoding_time/100)
            plot([0,0],[ax.get_ylim()[0],ax.get_ylim()[1]],linewidth=1,linestyle='-',color=(0,0,0))
            plot([1600,1600],[ax.get_ylim()[0],ax.get_ylim()[1]],linewidth=1,linestyle='--',color=(0.7,0.7,0.7))
            legend()

        plt.savefig(savePath, dpi=300)
        plt.close()
        
    '''  
    def SME_ripple_interaction(self):
        
        # ripple_exists array is a binary vector of shape num_events, with 1 indicating presence of a ripple
        ripple_idxs, _ = super().ripple_idxs_func()
        ripple_exists = np.zeros_like(self.word_correct_array_np)
        ripple_exists[ripple_idxs] = 1
        assert np.sum(ripple_exists) == ripple_idxs.shape[0], print("Something is not right with the ripple shapes.")
        
        # run mixed effects model 
        vc = {'session':'0+session'}

        SE_df = pd.DataFrame(data={'session':self.session_name_array_np,'subject':self.subject_name_array_np,'ripple_exists':ripple_exists, 
                                    'word_recalled': self.word_correct_array_np, 'HFA_mean': self.HFA_mean})
        get_bin_CI_model = smf.mixedlm("word_recalled ~ ripple_exists*HFA_mean", SE_df, groups="subject", vc_formula=vc, 
                                        re_formula='ripple_exists*HFA_mean')
        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)
        
        # run OLS model for verification 
        get_bin_CI_model_ols = smf.ols("word_recalled ~ ripple_exists*HFA_mean", SE_df)
        bin_model_ols = get_bin_CI_model_ols.fit()
            
        return bin_model, bin_model_ols
        
    def recalled_data(self):
        
        recalled = self.word_correct_array_np
        self.ripple_exists_re = self.ripple_exists_np[recalled==1]
        self.no_ripple_exists_re = 1 - self.ripple_exists_re
        self.clustered_re = self.semantic_clustered_array_np[recalled==1].squeeze()
        self.HFA_re = self.HFA_mean[recalled==1].squeeze()
        self.sess_re= self.session_name_array_np[recalled==1].squeeze()
        self.subj_re = self.subject_name_array_np[recalled==1].squeeze()
        
    def lmm_model(self, dep_var_list, formula_list, re_formula_list, recalled_bool_list, 
                  params_list, savePath):
        
        data = pd.DataFrame(data={'recalled':self.word_correct_array_np, 
                                        'ripple_exists':self.ripple_exists_np, 'HFA_mean':self.HFA_mean, 
                                        'subject':self.subject_name_array_np, 
                                        'session':self.session_name_array_np, 
                                        'serial_pos':self.serialpos_array_np})
        
        num_models = len(dep_var_list)
        vc = {'session':'0+session'}
        
        for i in range(num_models):
            
            formula = formula_list[i]
            re_formula = re_formula_list[i]
            
            model = smf.mixedlm(formula, data, groups="subject", vc_formula=vc, 
                                            re_formula=f'{re_formula}')
            
            model_fit = model.fit(method=["lbfgs"])
            
            print(model_fit.summary())
            
            self.save_model_info(model_fit, params_list[i], f'{savePath}_{formula}.csv')
            
        return model_fit
    
         
    def save_model_info(self, model, params, savePath):

        param_dict = {}
        param_dict['name'] = []
        param_dict['coef'] = []
        param_dict['stderr'] = []
        param_dict['pval'] = []
        param_dict['tval'] = []
        
        for param in params:

            param_dict['name'].append(param)
            param_dict['coef'].append(model.params[param])
            param_dict['stderr'].append(model.bse[param])
            param_dict['pval'].append(model.pvalues[param])
            param_dict['tval'].append(model.tvalues[param])

        pd.DataFrame(param_dict).to_csv(savePath, index=False)  

    
     def semantic_clustering(self, ripple_delta, HFA_delta):
        
        (Based on my current understanding)
        
        Input:
        
            :param bool rippleDelta: if True, use start ripple times to compute sess_delta
            :param bool hfa_delta: if True, use hfa array to compute sess_delta
        
        Output: 
        
            numpy array sess_delta: mean difference between ripple rates or hfa 
            for clustered vs unclustered recalled
        
        self.ripple_delta = ripple_delta
        self.HFA_delta = HFA_delta

        assert ripple_delta != HFA_delta, print("Boolean arguments should not be equal")
        
        self.num_sessions = 0 
        
        self.counter_delta = 0 
        
        # select which serialpositions you're looking at (since curious if 1-6 show all the SCE)
        serialpos_select = np.arange(1,13) 

        # EF1208, what is this?
        remove_chaining = 0 # 2022-07-19 trying a control to see if SCE still exists after removing recalls that begin with SP 1+2 in a row
        
        # these values are all for subject-level SCE v. avg_recalls analysis
        if self.sub_selection == 'whole':
            min_SCE_trials = 20 # minimum SCE trials in session to include in SCE v. avg_recalls plot
        else:
            min_SCE_trials = 10
            
        stats_bin = self.ripple_bin_start_end[1]-self.ripple_bin_start_end[0] # only using 1 bin for encoding 

        self.adj_semantic_encoding_array = []
        self.rem_semantic_encoding_array = []
        self.rem_unclustered_encoding_array = []
        self.last_recall_encoding_array = [] # the last word remembered on each list (no transitions)...but make sure it's not an intrusion or repeat too!
        self.forgot_encoding_array = []
        self.sub_name_array0 = []; self.sess_name_array0 = []; self.elec_name_array0 = []
        self.sub_name_array1 = []; self.sess_name_array1 = []; self.elec_name_array1 = []
        self.sub_name_array2 = []; self.sess_name_array2 = []; self.elec_name_array2 = []
        self.sub_name_array3 = []; self.sess_name_array3 = []; self.elec_name_array3 = []
        self.sess_name_array4 = [] # forgot why I keep the others but leaving them 2022-06-10
        self.sub_name_array5 = []; self.sess_name_array5 = []; self.elec_name_array5 = []

        # for clustered v. unclustered subject-level analysis (need to record at session-level though for mixed model)
        self.sess_sessions = []
        self.sess_delta = []
        self.sess_subjects = []
        self.sess_recall_num = []
        self.sess_clust_num = []
        self.sess_prop_semantic = []
        
        session_names = np.unique(self.session_name_array)
        
        # EF1208, converting to numpy for parallel indexing 
        for sess in session_names:
            
            self.num_sessions += 1
            
            # for each session will get a clustered and unclustered a) ripple start array, or b) HFA_array
            clustered_data = []; unclustered_data = []
            
            # and also the proportion of semantically clustered recalls
            temp_corr = []; temp_sem_key = []
            
            # Number of lists for a given session 
            # EF1208, on AMY first half I'm seeing a session with 10 lists, is that normal?
            sess_list_nums = np.unique(self.list_num_key_np[self.session_names_np==sess]) 
            
            # loop through each list in the session 
            for ln in sess_list_nums:
                
                # obtain electrodes corresponding to the selected list
                list_elec_array = np.unique(self.electrode_array_np[(self.session_names_np==sess) & (self.list_num_key_np==ln)])
                
                for elec in list_elec_array:
    
                    # boolean array, with the number of True elements equal to the list length (12)
                    list_ch_idxs = (self.session_names_np==sess) & (self.list_num_key_np==ln) & (self.electrode_array_np==elec) 
                    
                    if ripple_delta:
                        list_ch_encoding_array = self.start_array_np[list_ch_idxs] # ripple start times for the selected list 
                        single_event_time = self.ripple_bin_duration
                    elif HFA_delta:
                        list_ch_encoding_array = self.HFA_array_np[list_ch_idxs]
                        single_event_time = self.HFA_bin_duration
                        
                    list_ch_cats = self.cat_array_np[list_ch_idxs] # semantic category of presented words
                    list_ch_corr = self.word_correct_array_np[list_ch_idxs] # binary array, whether or not the word was correctly recalled 
                    list_ch_semantic_key = self.semantic_array_np[list_ch_idxs] # list of lists containing recall (A, C, D, Z) for each recalled word
                    list_ch_recall_positions = self.recall_position_np[list_ch_idxs] # list containing encoded position of recalled words
                    
                    # remove ones starting with serialpos 1->2 as a control (or just 1 if it's len 1)
                    if remove_chaining == 1:
                        if len(list_ch_recall_positions[0])==1:
                            if list_ch_recall_positions[0][0]==1: # if 1st serialpos
                                continue # get out of this loop if only one recall and it's serialpos 1
                        elif len(list_ch_recall_positions[0])>0:
                            if ((list_ch_recall_positions[0][0]==1)&(list_ch_recall_positions[0][1]==2)):     
                                continue # get out of loop if recalls are serialpos 1->2 (no matter what)            

                    for i_recall_type, recall_type in enumerate(list_ch_semantic_key[0]): # all 12 lists have same values so just take 1st one
                        recall_position = list_ch_recall_positions[0][i_recall_type] # ditto re: taking 1st
                        if recall_position in serialpos_select: 
                            if recall_type == 'A': # adjacent semantic and adjacent in time 
        #                     if recall_type in ['A','C']: # adjacent AND remote semantic
                                # note the -1 since recall positions are on scale of 1-12
                                self.adj_semantic_encoding_array = superVstack(self.adj_semantic_encoding_array, list_ch_encoding_array[recall_position-1])
                                self.sub_name_array0.append(sess[0:6])
                                self.sess_name_array0.append(sess)
                                self.elec_name_array0.append(elec)
                            elif recall_type == 'C': # remote semantic, remote in time but from the same semantic category 
                                self.rem_semantic_encoding_array = superVstack(self.rem_semantic_encoding_array,list_ch_encoding_array[recall_position-1])
                                self.sub_name_array1.append(sess[0:6])
                                self.sess_name_array1.append(sess)
                                self.elec_name_array1.append(elec)
                            elif ( (recall_type == 'D') ): # & (recall_position>0) ): # remote unclustered
                                self.rem_unclustered_encoding_array = superVstack(self.rem_unclustered_encoding_array,list_ch_encoding_array[recall_position-1])
                                self.sub_name_array2.append(sess[0:6])
                                self.sess_name_array2.append(sess)  
                                self.elec_name_array2.append(elec)
                            elif ( (recall_type == 'Z') ): #& (recall_position>0) ): # last word of list & was actually a recalled word
                                self.last_recall_encoding_array = superVstack(self.last_recall_encoding_array,list_ch_encoding_array[recall_position-1])
                                self.sub_name_array3.append(sess[0:6])
                                self.sess_name_array3.append(sess)
                                self.elec_name_array3.append(elec)
                            else:
                                self.sess_name_array4.append(sess[0:6])
                                
                        # Creating clustered vs unclustered conditioned 
                        # start_arracyC will be of shape N x ripple_start:ripple_end, where N is the number of clustered recalls
                        # and ripple_start:ripple_end is the timestep range of interest for ripples
                        # same goes for start_arrayU, except N is the number of unclustered recalled 
                        if recall_position in serialpos_select: # so can select by serialpos (e.g. 1:6 or 7:12)
                            if recall_type in ['A','C']: # adjacent semantic or remote semantic
                                # note the -1 since recall positions are on scale of 1-12
                                clustered_data = superVstack(clustered_data,list_ch_encoding_array[recall_position-1])
                            elif ( (recall_type in ['D','Z']) & (recall_position>0) ): # remote unclustered or dead end (>0 means recalled word)
                                unclustered_data = superVstack(unclustered_data,list_ch_encoding_array[recall_position-1])
                                
                                
                    # unpack semantic clustering key to trial level (only need to do once for one electrode)
                    if elec == list_elec_array[0]:
                        for word in range(sum(list_ch_idxs)): 
                            if (word+1) in list_ch_recall_positions[0]: # serial positions are 1-indexed so add 1 to check in list_ch_recall_positions
                                temp_corr.append(1)
                                # use index from serialpos to get clustering classification
                                if ((sess== 'R1108J-2')&(ln==25)): # single mistake showss up
                                    if word == 8:
                                        temp_sem_key.append('A')
                                    elif word == 9:
                                        temp_sem_key.append('Z')
                                else: 
                                    temp_sem_key.append(list_ch_semantic_key[0][list_ch_recall_positions[0].index(word+1)])
                            else:
                                temp_corr.append(0)
                                temp_sem_key.append('')               
                                
                    # make forgotten array to plot along with SCE too which is easy enough 
                    forgotten_words = 1-np.array(list_ch_corr)
                    if sum(forgotten_words)>0: # R1065 a whiz
                        self.forgot_encoding_array = superVstack(self.forgot_encoding_array,np.array(list_ch_encoding_array)[findInd(forgotten_words),:])
                        self.sub_name_array5.extend(np.tile(sess[0:6],int(sum(forgotten_words))))
                        self.sess_name_array5.extend(np.tile(sess,int(sum(forgotten_words))))
                        self.elec_name_array5.extend(np.tile(elec,int(sum(forgotten_words))))
                        
                        
            if ((len(clustered_data)>min_SCE_trials) & (len(unclustered_data)>min_SCE_trials) & \
                (len(clustered_data)!=single_event_time) ): # last one in there for a len(1) start_arrayC

                # back at session-level record the delta, sub, sess, and avg_recall_num for *all* trials
                self.sess_sessions.append(sess)
                self.sess_subjects.append(sess[0:6])   

                # can just use list_elec_array to select only one electrode we know exists for this session (altho should be irrelevant when we average anyway)
                # This is a binary array, where a 1 indicates correct recall of a word 
                sess_word_correct_array = self.word_correct_array_np[((self.electrode_array_np==list_elec_array[0]) & (self.session_name_array_np==sess))]
                
                
                self.sess_recall_num.append(self.list_length*(sum(sess_word_correct_array)/len(sess_word_correct_array)))
                
                self.sess_prop_semantic.append(sum([trial in ['A','C'] for trial in temp_sem_key])/sum(temp_corr))
                
                
                # create histogram for ripples/HFA associated with clustered + unclustered words
                binned_clustered_data = binBinaryArray(clustered_data,stats_bin,self.sr_factor)
                binned_unclustered_data = binBinaryArray(unclustered_data ,stats_bin,self.sr_factor)
                
                # take difference between histogram means 
                self.sess_delta.append(np.mean(binned_clustered_data)-np.mean(binned_unclustered_data)) 
      
        trial_nums = [len(self.sub_name_array0),len(self.sub_name_array1),len(self.sub_name_array2),len(self.sub_name_array3),len(self.sess_name_array4)]
        
        
    def plot_SME_or_SCE_HFA(self, analysis_type, mode, title_str, savePath):

        :param int analysis_type: 0 for ripple/no ripple, 1 for early list/middle list, 2 for all data, 
            3 for clustered/unclustered (only recalled), 4 for clustered/unclustered for ripple/no ripple group
        :param int mode: 
            if analysis type is set to 0 or 4, then mode can be 0 for only ripples, and 1 for only non-ripples
            if analysis type is set to 1, then mode can be 0 for using early list items, and 1 for using middle list items
            for analysis type set to 2 and 3, mode is not applicable  
     
        pad = int(np.floor(self.smoothing_triangle/2)) 
        
        # text for plot label
        cat_zero_text = 'Recalled'
        cat_one_text = 'Not recalled'
        
        # ripple/no ripple
        if analysis_type == 0:
            HFA_array, partition_array, subject_name_array, session_name_array, _ = super().separate_HFA_by_ripple(mode)
        # early/middle 
        elif analysis_type == 1:
            HFA_array, partition_array, subject_name_array, session_name_array, _ = super().separate_HFA_by_serial_pos(mode)
        # all data
        elif analysis_type == 2:
            HFA_array = self.HFA_array_np
            partition_array = self.word_correct_array_np
            subject_name_array = self.subject_name_array_np
            session_name_array = self.session_name_array_np
        # clustered vs unclustered (only recalled)
        elif analysis_type == 3:
            cat_zero_text = 'Clustered'
            cat_one_text = 'Unclustered'
            HFA_array = self.HFA_array_np
            partition_array = self.semantic_clustered_array_np
            subject_name_array = self.subject_name_array_np
            session_name_array = self.session_name_array_np
        elif analysis_type == 4:
            cat_zero_text = 'Clustered'
            cat_one_text = 'Unclustered'
            
            # divide into ripple/no ripple group
            HFA_array, word_correct_array, subject_name_array, session_name_array, selected_idxs = super().separate_HFA_by_ripple(mode)
            clustered_array_selected = self.semantic_clustered_array_np[selected_idxs]
            
            # now select only recalled words for clustered/unclustered analyses
            correct_idxs = np.argwhere(word_correct_array==1)
            HFA_array = np.squeeze(HFA_array[correct_idxs])
            subject_name_array = np.squeeze(subject_name_array[correct_idxs])
            session_name_array = np.squeeze(session_name_array[correct_idxs])
            partition_array = np.squeeze(clustered_array_selected[correct_idxs])
            
        else:
            print("Analysis type can only be 0 (ripple/noripple), 1 (early/middle), or 2 (all data), or 3 (clustered), \
                  or 4 (clustered for ripple/noripple)")
            return 0
            
        plot_ME_mean = 1 # 0 for typical PSTH, 1 for ME mean, 2 for average across sub averages

        # set up the PVTH stats parameters here too (for encoding have 30 bins)
        psth_start = int(self.pre_encoding_time/self.bin_size)
        psth_end = int(self.encoding_time/self.bin_size)

        bin_centers = np.arange(psth_start+0.5,psth_end)
        xr = bin_centers #np.arange(psth_start,psth_end,binsize)

        # get vectors of encoding list identifier data for forgotten and recalled words
        # in encoded_word_key_array, 0 for not recalled, 1 for recalled, 2 for recalled 
        # but was an IRI<2 (don't care about that for encoding)
        start_array_enc_0 = HFA_array[partition_array==0] # not recalled/ unclustered (depending on analysis_type)
        start_array_enc_1 = HFA_array[partition_array==1] # recalled/clustered (depending on analysis_type)

        # same for sub and sess
        sub_0 = subject_name_array[partition_array==0]
        sess_0 = session_name_array[partition_array==0]
        sub_1 = subject_name_array[partition_array==1]
        sess_1 = session_name_array[partition_array==1]

        PSTH_all = triangleSmooth(np.mean(HFA_array,0),self.smoothing_triangle)
        
        # for ripple/no ripple and early/middle analyses
        # ensure that y axis is the same for both conditions
        if analysis_type == 0 or analysis_type == 1 or analysis_type == 4:
            if analysis_type == 0 or analysis_type == 4:
                HFA_array_0, _, _, _, _ = super().separate_HFA_by_ripple(0)
                HFA_array_1, _, _, _, _ = super().separate_HFA_by_ripple(1)
            else:
                HFA_array_0, _, _, _, _ = super().separate_HFA_by_serial_pos(0)
                HFA_array_1, _, _, _, _ = super().separate_HFA_by_serial_pos(1)
                
            PSTH_all_0 = triangleSmooth(np.mean(HFA_array_0,0),self.smoothing_triangle)
            PSTH_all_1 = triangleSmooth(np.mean(HFA_array_1,0),self.smoothing_triangle)
            min_val_PSTH_0 = np.min(PSTH_all_0)
            max_val_PSTH_0 = np.max(PSTH_all_0)
            min_val_PSTH_1 = np.min(PSTH_all_1)
            max_val_PSTH_1 = np.max(PSTH_all_1)
            max_val_PSTH = max(max_val_PSTH_0, max_val_PSTH_1) + .3
            min_val_PSTH = min(min_val_PSTH_0, min_val_PSTH_1) - .3 
        
        # don't need to worry about this for all data since there's only one condition 
        else:
            min_val_PSTH = np.min(PSTH_all)
            max_val_PSTH = np.max(PSTH_all)
            
    '''
        
        