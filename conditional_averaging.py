def cond_av(S, T, smin=None, smax=None, Sref=None, prominence=None, delta=None, overlap=True, discard_small = False,
             vspace = None, weight='amplitude'):
    """
    Use: cond_av(S, T, smin, smax=None, Sref=None, delta=None, overlap=True)
    Use the level crossing algorithm to compute the conditional average of
    a process.
    Inputs:
        S: Signal. Size N ................................. (1xN) np array
        T: Time base ...................................... (1xN) np array
        smin: Minimal peak amplitude
              in units of rms-value above mean value. ..... Either float, 
              None or (1xN) np array. def None
        smax: Maximal peak amplitude. ..................... Either float, 
        None or (1xN) np array. def None
        Sref: Reference signal.
              If None, S is the reference. ................ (1xN) np array,
                                                            def None
        delta: The size of the conditionally averaged signal. If overlap = False,
               it is also the minimal distance between two peaks.
               If delta = None, it is estimated as
               delta = len(S)/(number of conditional events)*timestep.
               ............................................ float, def None
        overlap: Overlapping of averaging windows. If False, windows are not
                 allowed to overlap, no part of the signal is included multiple
                 times and delta also gives the minimal distance between peaks.
                ........................................... bool, def True
        discard_small: If True, peaks where another part of the signal, within
                        one delta of the peak, is larger that the peak itself are
                        discarded.......................... bool, def False
        prominence: Minimal peak prominence in units
                    of rms-value above mean value.......... Either a number, 
                    None, an array matching x or a 2-element sequence of the 
                    former. The first element is always interpreted as the 
                    minimal and the second, if supplied, as the maximal required 
                    prominence. def None
        vspace: Required vertcal height of peaks, the vertical distance to its 
                   neighboring samples. ................... Either a number, 
                   None, an array matching x or a 2-element sequence of the 
                   former. The first element is always interpreted as the 
                   minimal and the second, if supplied, as the maximal required
                   threshold. ............................. def None
        weight: Weighting to be used in the conditionally averaged signal. If
                weight='amplitude' the amplitudes of each peak decides its 
                weight in the average. If weight='equal' all peaks are 
                normalized by either peak height or prominence before averaging
                ........................................... str, def 'amplitude'
    Outputs:
        Svals: Signal values used in the conditional average.
               S with unused values set to nan. ........... (1xN) np array
        s_av: Conditionally averaged signal ............... np array
        s_var: Conditional variance of events ............. np array
        t_av: Time base of s_av ........................... np array
        peaks: Max amplitudes of conditionally averaged events.
        ................................................... np array
        prominences: Prominence values of conditionally averaged events
        ................................................... np array
        wait: Waiting times between peaks. ................ np array
    """
    
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences
    
    if all(i is None for i in[smin, prominence]):
        raise TypeError('Missing 1 required positional argument: \'smin\' '
                        'or \'prominence\'')
    
    if Sref is None:
        Sref = S.copy()
    assert len(Sref) == len(S) and len(S) == len(T)

    sgnl = (Sref - np.mean(Sref)) / np.std(Sref)
    dt = np.diff(T).sum() / (len(T) - 1)
    
    #Estimating delta.
    if delta is None:
        if smin is None:
            tmpmin = prominence
        else:
            tmpmin = smin
            
        places = np.where(sgnl > tmpmin)[0]
        dplaces = np.diff(places)
        split = np.where(dplaces != 1)[0]
        # (+1 since dplaces is one ahead with respect to places)
        lT = np.split(places, split + 1)
        delta = len(sgnl) / len(lT) * dt
    
    distance = None
    # Ensure distance delta between peaks.
    if not overlap:
        distance = int(delta / dt)   
     
    
    # Find peak indices.
    gpl_array, properties = find_peaks(sgnl, height = [smin, smax], distance = distance,
                              prominence = prominence, threshold = vspace)
    
    if not prominence and overlap:
        places = np.where(sgnl > smin)[0]
        if smax:
            higher = np.where(sgnl<smax)[0]
            places = np.intersect1d(places, higher)
        assert len(places) > 0, "No conditional events"
   
        dplaces = np.diff(places)
        split = np.where(dplaces != 1)[0]
        lT = np.split(places, split + 1)
        #Ensure one peak for every threshold crossing. Largest peak is chosen.
        #Ties are broken my earliest time.
        for i, lTi in enumerate(lT): 
            peak_check = np.isin(gpl_array, lTi)
            if peak_check.sum()>1:
                peak_ind = gpl_array[peak_check]
                highest_local_peak = Sref[peak_ind].argmax()
                not_highest_local_peaks = np.delete(peak_ind, highest_local_peak)
                gpl_array = np.delete(gpl_array, np.isin(gpl_array, not_highest_local_peaks))
        
    
    # Use arange instead of linspace to guarantee 0 in the middle of the array.
    t_av = np.arange(-int(delta / (dt * 2)), int(delta / (dt * 2)) + 1) * dt

    # For use in the individual normalizations.
    prominences = peak_prominences(S, gpl_array)[0]

    Svals = np.zeros(len(sgnl))
    Svals[:] = np.nan

    badcount = 0
    gpl_array= gpl_array.astype(float)

    t_half_len = int((len(t_av) - 1) / 2)
    s_tmp = np.zeros([len(t_av), len(gpl_array)])
    


    # Taking equally sized signal excerpts around every peak and storing them in an array.
    for i, global_peak_loc in enumerate(gpl_array):

        # Setting up an array of all the signal values around the conditional events.
        low_ind = int(max(0, global_peak_loc - t_half_len))
        high_ind = int(min(len(sgnl), global_peak_loc + t_half_len + 1))
        tmp_sn = S[low_ind:high_ind].copy()
             
        if low_ind == 0:
            tmp_sn = np.append(np.zeros(-int(global_peak_loc) + t_half_len), tmp_sn)
        if high_ind == len(S):
            tmp_sn = np.append(
                tmp_sn, np.zeros(int(global_peak_loc) + t_half_len + 1 - len(S))
            )
            
        # If discard_small=True only peak values which are the max value within one delta of
        # the peak location are included in the average.
        if discard_small:
            # Making and checking a window within the reference signal to see if the the peak
            # is the local maximum
            tmp_snref = sgnl[low_ind:high_ind].copy()
            
            if low_ind == 0:
                tmp_snref = np.append(np.zeros(-int(global_peak_loc) + t_half_len), tmp_snref)
            if high_ind == len(S):
                tmp_snref = np.append(
                    tmp_snref, np.zeros(int(global_peak_loc) + t_half_len + 1 - len(S))
                )
                
            if tmp_snref.max() != tmp_snref[t_half_len]:
            
                tmp_sn[:] = np.nan
                gpl_array[i] = np.nan
            
                badcount += 1
        
        if not np.isnan(gpl_array[i]):
            Svals[low_ind:high_ind] = S[low_ind:high_ind].copy()
            
            
        
        s_tmp[:, i] = tmp_sn 
        
        # Normalizing each conditional event by its peak or prominence value. If both threshold
        # conditions are used the peak value is the normalizing factor.
        if weight == 'equal' and not np.isnan(tmp_sn.min()):
            if smin is None:
                s_tmp[:, i] /= prominences[i]
                s_tmp[:, i] += (1 - s_tmp[t_half_len, i])
                
            else:
                s_tmp[:, i] /= tmp_sn[t_half_len]
                
        
                
    # Removing NaNs from the peak indicies array which stems from the discard_small=True condition.
    gpl_array = gpl_array[~np.isnan(gpl_array)].astype(int)
    
    # The peak values, prominence values and waiting times for each peak.
    peaks = S[gpl_array]
    prominences = peak_prominences(S, gpl_array)[0]
    wait = np.append(np.array([T[0]]), T[gpl_array])
    wait = np.diff(wait)
                
    # The average of all signal excerpts. Referred to as the conditionally averaged signal or the
    # conditionally averaged waveform.
    s_av = np.nanmean(s_tmp, axis=1)

    # The conditional variance of the conditional event f(t) is defined as
    # CV = <(f-<f>)^2>/<f^2> = 1 - <f>^2/<f^2>
    # at each time t.
    # For a highly reproducible signal, f~<f> and CV = 0.
    # For a completely random signal, <f^2> >> <f>^2 and CV = 1.
    # OBS: We return 1-CV = <f>^2/<f^2>.
    s_var = s_av ** 2 / np.nanmean(s_tmp ** 2, axis=1)
    
    
    print("conditional events:{}".format(len(peaks)), flush=True)
    if discard_small and badcount > 0:       
        print(f"Removed bursts where the recorded peak was not the largest:{badcount}.")


    return Svals, s_av, s_var, t_av, peaks, wait, prominences
