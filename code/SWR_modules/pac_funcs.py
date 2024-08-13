def permute_amplitude_time_series(amplitude_time_series):
    # Get the length of the time series
    series_length = len(amplitude_time_series)

    # Choose a random index to cut the time series
    cut_index = np.random.randint(1, series_length)

    # Create the permuted time series by reversing the order of both parts
    permuted_series = np.concatenate((amplitude_time_series[cut_index:], amplitude_time_series[:cut_index]))

    return permuted_series

def compute_entropy(data_arr):
    import math
    probability_arr = (data_arr/np.sum(data_arr))
    probability_arr = probability_arr[probability_arr!=0]
    negative_vals = probability_arr[probability_arr < 0]
    if negative_vals.shape[0] > 0:
        print(probability_arr)
    if np.all(np.isnan(probability_arr)):
        print(probability_arr)
    entropy = -np.sum(probability_arr*np.log(probability_arr))
    return entropy

def compute_pac_p_value(entropy_vals):
    
    '''
    :param list entropy_vals: entropy of PAC histogram, with first value 
    indicating PAC of non-permuted data
    '''
    
    p_value = 1 - np.argwhere(entropy_vals[0] < entropy_vals[1:]).shape[0]/len(entropy_vals)
    return p_value