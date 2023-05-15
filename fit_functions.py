import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as sst
import scipy.special as ssp

def dexp(peak, t, rise, fall, A):
    '''
    function description
    
    '''
    pulse = (1-A) * np.ones(len(t))
    if rise == 0:
        pulse[peak:] = A * np.exp(-t[peak:] / fall) + (1-A)
    elif fall == 0:
        pulse[:peak] = A * np.exp(t[:peak] / rise) + (1-A)
    else:
        pulse[peak:] = A * np.exp(-t[peak:] / fall ) + (1-A)
        pulse[:peak] = A * np.exp(t[:peak] / rise) + (1-A)
    return pulse

def dexp_pure(peak, t, rise, fall, A, b):
    '''
    function description
    
    '''
    pulse = b * np.ones(len(t))
    if rise == 0:
        pulse[peak:] = A * np.exp(t[peak:] / fall) + b
    elif fall == 0:
        pulse[:peak] = A * np.exp(t[:peak] / rise) + b
    else:
        pulse[peak:] = A * np.exp(t[peak:] / fall ) + b
        pulse[:peak] = A * np.exp(t[:peak] / rise) + b
    return pulse

def lorentz(peak, t, duration, A):
    pulse = A * ((1 + ((t-t[peak]) / duration) ** 2) ** (-1)) + (1-A)
    return pulse

def gaussian(peak, t, duration, A):
    pulse = A * np.exp(-((t-t[peak]) / duration)**2 / 2) + (1-A)
    return pulse

def triangular(peak, t, duration, A):
    pulse = (1-A) * np.ones(len(t))
    lower = abs(t-t[peak]) < duration
    pulse[lower] =  A * (1 - abs(t[lower]-t[peak]) / duration) + (1-A)
    return pulse

def box(peak, t, duration, A):
    pulse = (1-A) * np.ones(np.size(t))
    pulse[np.absolute(t-t[peak]) < duration / 2] = A + (1-A)
    return pulse

def gammad(peak, t, alpha, duration, A):
    pulse = (1-A) * np.ones(len(t))
    pulse[t+alpha-1-t[peak]>= 0] = A * ((t[t +alpha-1-t[peak]>= 0] + alpha-1- t[peak])/ duration) ** (alpha-1) * np.exp(-(t[t + alpha-1- t[peak]>= 0]+alpha-1- t[peak]) / 
                                                                              duration) / (((alpha-1)/duration)**(alpha-1)*np.exp(-(alpha-1)/duration))+ (1-A)
    return pulse

def rayleigh(peak, t, duration, A):
    pulse = (1-A) * np.ones(len(t))
    pulse[t+1-t[peak]>= 0] = (1 / (np.exp(-1**2 / 2))) * A * ((t[t +1-t[peak]>= 0] + 1- t[peak])/ duration) * np.exp(-((t[t + 1- t[peak]>= 0]+1- t[peak]) / duration)**2 / 2) + (1-A)
    return pulse

def fit_dexp(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return dexp(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (0.5, 0.5, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    estimates = [estimate[0] / (estimate[0] + estimate[1]), estimate[0] + estimate[1], estimate[2]]
    
    # Make a realization by using the estimates.
    realized_estimate = dexp(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimates, error

def fit_gaussian(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return gaussian(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (1, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    
    # Make a realization by using the estimates.
    realized_estimate = gaussian(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimate, error

def fit_lorentz(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return lorentz(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (1, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    
    # Make a realization by using the estimates.
    realized_estimate = lorentz(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimate, error

def fit_triangular(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return triangular(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (1, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    
    # Make a realization by using the estimates.
    realized_estimate = triangular(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimate, error

def fit_box(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return box(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (0.5, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    
    # Make a realization by using the estimates.
    realized_estimate = box(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimate, error

def fit_gammad(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return gammad(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (2, 1, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    
    # Make a realization by using the estimates.
    realized_estimate = gammad(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimate, error


def fit_rayleigh(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return rayleigh(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (1, s_av[peak])
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    
    # Make a realization by using the estimates.
    realized_estimate = rayleigh(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimate, error


def fit_dexp_pure(t_av, s_av):
    '''
    function description
    
    '''
    # Determine peak in the conditionally averaged pulse.
    peak = np.argmax(s_av)
    
    # The function to be optimized against the data.
    def fit_template(*args):
        return dexp_pure(peak, *args)
    
    # Initial guess of parameters.
    initial_guess = (0.5, -0.5, s_av[peak], 0)
    
    # Fit to the double exponential with least squares.
    estimate, cov = curve_fit(fit_template, t_av, s_av, p0=initial_guess)
    estimates = [estimate[0] / (estimate[0] - estimate[1]), estimate[0] - estimate[1], estimate[2], estimate[3]]
    
    # Make a realization by using the estimates.
    realized_estimate = dexp_pure(peak, t_av, *estimate)
    
    # Standard error.
    error = np.sqrt(np.diag(cov))
    
    return t_av, realized_estimate, estimates, error

def Apdf_exp_fit(data, mean, std, method):
    if method == 's':
        threshold = 2.5*std+mean
    elif method == 'p':
        threshold = 2.5*std
    _, beta = sst.expon.fit(data - threshold, floc = 0)
    return beta

def Wpdf_exp_fit(data, window=None):
    flocT = 0
    if window:
        flocT = None
    _, beta = sst.expon.fit(data, floc = flocT)
    return beta