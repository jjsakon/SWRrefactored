import numpy as np
    
def triangleSmooth(data,smoothing_triangle): # smooth data with triangle filter using padded edges
    
    # problem with this smoothing is when there's a point on the edge it gives too much weight to 
    # first 2 points (3rd is okay). E.g. for a 1 on the edge of all 0s it gives 0.66, 0.33, 0.11
    # while for a 1 in the 2nd position of all 0s it gives 0.22, 0.33, 0.22, 0.11 (as you'd want)
    # so make sure you don't use data in first 2 or last 2 positions since that 0.66/0.33 is overweighted
    
    factor = smoothing_triangle-3 # factor is how many points from middle does triangle go?
    # this all just gets the triangle for given smoothing_triangle length
    f = np.zeros((1+2*factor))
    for i in range(factor):
        f[i] = i+1
        f[-i-1] = i+1
    f[factor] = factor + 1
    triangle_filter = f / np.sum(f)

    padded = np.pad(data, factor, mode='edge') # pads same value either side
    smoothed_data = np.convolve(padded, triangle_filter, mode='valid')
    return smoothed_data
        
def generate_SCE_SME_plots(self, start_array_enc_1, start_array_enc_0, sub_1, sess_1, sub_0, sess_0, 
                            cat_one_text, cat_zero_text, ylabel_str, plot_ME_mean, bin_centers, title_str, 
                            min_val_PSTH, max_val_PSTH, savePath, smoothing_triangle=5):
    
    pad = int(np.floor(smoothing_triangle/2)) 
    
    # loop through recalled and then forgotten words
    for category in range(2):
        
        if category == 0:
            temp_start_array = start_array_enc_1
            sub_name_array = sub_1
            sess_name_array = sess_1

            # HFA 
            PSTH = triangleSmooth(np.mean(temp_start_array,0),smoothing_triangle)
            subplots(1,1,figsize=(5,4))
            plot_color = (0,0,1)
            num_words = f"{cat_zero_text}: {temp_start_array.shape[0]}"

        else:       
            temp_start_array = start_array_enc_0
            sub_name_array = sub_0
            sess_name_array = sess_0
            
            PSTH = triangleSmooth(np.mean(temp_start_array,0),smoothing_triangle)

            plot_color = (0,0,0)
            num_words = f"{cat_one_text}: {temp_start_array.shape[0]}"

        # note that output is the net ± distance from mean
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")    
            mean_plot,SE_plot = getMixedEffectMeanSEs(temp_start_array,sub_name_array,sess_name_array)

        if plot_ME_mean == 1:
            PSTH = triangleSmooth(mean_plot,smoothing_triangle) # replace PSTH with means from ME model (after smoothing as usual)  
        elif plot_ME_mean == 2:
            temp_means = []
            for sub in np.unique(sub_name_array):
                temp_means = superVstack(temp_means,np.mean(temp_start_array[np.array(sub_name_array)==sub],0))
            PSTH = triangleSmooth(np.mean(temp_means,0),smoothing_triangle)   
        
        xr = bin_centers #np.arange(psth_start,psth_end,binsize)
        if pad > 0:
            xr = xr[pad:-pad]
            binned_start_array = temp_start_array[:,pad:-pad] # remove edge bins    
            PSTH = PSTH[pad:-pad]
            SE_plot = SE_plot[:,pad:-pad]        
        
        plot(xr,PSTH,color=plot_color, label=num_words)
        fill_between(xr, PSTH-SE_plot[0,:], PSTH+SE_plot[0,:], alpha = 0.3)
        xticks(np.arange(pre_encoding_time+pad*100,encoding_time-pad*100+1,500)/100,
            np.arange((pre_encoding_time+pad*100)/1000,(encoding_time-pad*100)/1000+1,500/1000))
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
        ax.set_xlim(pre_encoding_time/100,encoding_time/100)
        plot([0,0],[ax.get_ylim()[0],ax.get_ylim()[1]],linewidth=1,linestyle='-',color=(0,0,0))
        plot([1600,1600],[ax.get_ylim()[0],ax.get_ylim()[1]],linewidth=1,linestyle='--',color=(0.7,0.7,0.7))
        legend()
        
    plt.plot()
