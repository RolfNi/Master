from .conditional_averaging import *
from .fit_functions import *
from fppanalysis import cond_av as OLDcond_av
def analyse(T, S, smin=None, smax=None, Sref=None, delta=None, window=None, prominence=None, weight='equal', method = 's'):
    Svals, s_av, s_var, t_av, peaks, wait, prominences = cond_av(S, T, smin = smin, smax = smax, Sref = Sref, delta = delta, prominence = 
                                                    prominence, window = window, weight = weight)
# =============================================================================
#     elif method == 'old':
#         Svals, s_av, s_var, t_av, peaks, wait = OLDcond_av(S, T, smin = smin, smax = smax, Sref = Sref, delta = delta,
#                                                            window = window)
# =============================================================================
    t_av, realized_estimate, estimates, error = fit_dexp(t_av, s_av)
    lmda, tau = estimates[0:2]
    
    if method=='s':
        Abeta = Apdf_exp_fit(peaks, S.mean(), S.std(), method)
    elif method == 'p':
        Abeta = Apdf_exp_fit(prominences, S.mean(), S.std(), method)
    
    Wbeta = Wpdf_exp_fit(wait[1:], window=window)
    return lmda, tau, Abeta, Wbeta

def amplitude_analyse(T, S, smin=None, smax=None, Sref=None, delta=None, window=False, prominence=None, weight='equal', method = 's'):
    Svals, s_av, s_var, t_av, peaks, wait, prominences = cond_av(S, T, smin = smin, smax = smax, Sref = Sref, delta = delta, prominence = 
                                                    prominence, window = window, weight = weight)
    if method == 's':
        beta = pdf_exp_fit(peaks, S.mean(), S.std(), method)
    elif method == 'p':
        beta = pdf_exp_fit(prominences, S.mean(), S.std(), method)
    
    return beta