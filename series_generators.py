import numpy as np
import scipy.signal as ssi
import model.point_model as pm
import model.pulse_shape as ps
import model.forcing as frc
import scipy.stats as sst
from tools.fit_functions import dexp

class MyFancyForcingGenerator(frc.ForcingGenerator):
    def __init__(self, amplitude_distribution, duration_distribution, wait_dist=None):
        self.wait_dist = wait_dist
        self.amplitude_distribution = amplitude_distribution
        self.duration_distribution = duration_distribution
        
        pass

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        
        if self.wait_dist is not None:
            dt = times[1]-times[0]
            waiting_times = self.wait_dist.rvs(size = int(times.max()*gamma*10))
            arrival_times = np.cumsum(waiting_times)
            times_above_max_indx = np.where(arrival_times>times.max())
            arrival_times = np.delete(arrival_times, times_above_max_indx)
            waiting_times = np.delete(waiting_times, times_above_max_indx)
            arrival_time_indx = (arrival_times / dt).astype(int)
            total_pulses = len(arrival_times)
            
        else:
            total_pulses = int(times.max() * gamma)
            arrival_time_indx = np.random.randint(0, len(times), size=total_pulses)
            
        
        amplitudes = self.amplitude_distribution.rvs(size = total_pulses)
        durations = self.duration_distribution.rvs(size = total_pulses)
        
        
        if self.wait_dist is not None:
            gamma = durations.mean() / waiting_times.mean()
        
        #print('Gamma from N samples:', gamma ,'Gamma from thorethical mean:', 1 / wait_dist.mean())
        
        return frc.Forcing(
            total_pulses, times[arrival_time_indx], amplitudes, durations
        )

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function,
    ):
        pass

    def set_duration_distribution(self, duration_distribution_function):
        pass
    

class PartialExp:
    def __init__(self, a, b, scale):
        self.a = a
        self.b = b
        self.scale = scale
        
    def rvs(self, size):
        if self.b == np.inf:
            return sst.expon.rvs(loc = self.a, scale = self.scale, size = size)

        rvs = np.zeros(size)
        start = 0
        stop = 0
        
        while True:
            
            rtest = sst.expon.rvs(loc = self.a, scale = self.scale, size = size)
            rtest = rtest[rtest<=self.b]
            stop = start + len(rtest)
            
            if stop >= size:
                stop = size
                rvs[start:size] = rtest[0:len(rvs[start:size])]
                break
            else:
                rvs[start:stop] = rtest

            start = stop
            
        return rvs
    
def gen_mixed_lams(gamma, T, dt, intervals):
    A0 = PartialExp(intervals[5], np.inf, 1)
    A01 = PartialExp(intervals[4], intervals[5], 1)
    A02 = PartialExp(intervals[3], intervals[4], 1)
    A03 = PartialExp(intervals[2], intervals[3], 1)
    A04 = PartialExp(intervals[1], intervals[2], 1)
    A05 = PartialExp(intervals[0], intervals[1], 1)
    
    def F(a,b):
        if b == np.inf:
            return np.exp(-a)
        else:
            return np.exp(-a)-np.exp(-b)

    dur = sst.rv_discrete(values = (1, 1))
    
    t0, s0 = gen_custom_exp_series_W(gamma * F(intervals[5], np.inf), T, dt, lam=0, duration_dist = dur, amplitude_dist = A0, wait_dist=None)
    t01, s01 = gen_custom_exp_series_W(gamma * F(intervals[4], intervals[5]), T, dt, lam=0.1, duration_dist = dur, amplitude_dist = A01, wait_dist=None)
    t02, s02 = gen_custom_exp_series_W(gamma * F(intervals[3], intervals[4]), T, dt, lam=0.2, duration_dist = dur, amplitude_dist = A02, wait_dist=None)
    t03, s03 = gen_custom_exp_series_W(gamma * F(intervals[2], intervals[3]), T, dt, lam=0.3, duration_dist = dur, amplitude_dist = A03, wait_dist=None)
    t04, s04 = gen_custom_exp_series_W(gamma * F(intervals[1], intervals[2]), T, dt, lam=0.4, duration_dist = dur, amplitude_dist = A04, wait_dist=None)
    t05, s05 = gen_custom_exp_series_W(gamma * F(intervals[0], intervals[1]), T, dt, lam=0.5, duration_dist = dur, amplitude_dist = A05, wait_dist=None)
    
    t, S = t0, s0+s01+s02+s03+s04+s05
    
    return t, S
    
    
    
def gen_custom_exp_series_W(gamma, T, dt, lam, duration_dist, amplitude_dist, wait_dist, noise_type = None, noise_to_signal_ratio = None):
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    pulse = ps.ExponentialShortPulseGenerator(lam)
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(MyFancyForcingGenerator(amplitude_distribution = amplitude_dist, duration_distribution=duration_dist, wait_dist=wait_dist))
    if wait_dist is not None:
        gamma = duration_dist.mean()/wait_dist.mean()
        
    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
    
    return t, S


def gen_custom_gaussian_series_W(gamma, T, dt, duration_dist, amplitude_dist, wait_dist, noise_type = None, noise_to_signal_ratio = None):
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    pulse = ps.GaussianShortPulseGenerator()
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(MyFancyForcingGenerator(amplitude_distribution = amplitude_dist, duration_distribution=duration_dist, wait_dist=wait_dist))
    if wait_dist is not None:
        gamma = duration_dist.mean()/wait_dist.mean()
        
    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
    
    return t, S


def gen_exp_series(gamma, T, dt, lam, duration, noise_type = None, noise_to_signal_ratio = None):
    if gamma == 'inf':
        return exp_convolution(T, dt, lam) 

    else:   
        series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
        pulse = ps.ExponentialShortPulseGenerator(lam)
        custom_forcing = pm.StandardForcingGenerator()
        custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
        series.set_pulse_shape(pulse)
        series.set_custom_forcing_generator(custom_forcing)
    
        if noise_type is not None:
            series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
            
        t, S = series.make_realization()
                
        return t, S


def exp_convolution(T, dt, lam):
    events = np.random.default_rng().standard_normal(int(T/dt))
    cutoff = -int(np.log(1e-50) / dt)
    pulsedomain = np.arange(-cutoff, cutoff + 1) * dt
    pulse = dexp(int(len(pulsedomain)/2), pulsedomain, lam, 1-lam, 1)
    #the direct method because fft- and oaconvolve did not play well on the fusion server...
    series = ssi.fftconvolve(events, pulse, mode='same') 
    t = np.linspace(0, T, int(T / dt))
    
    return t, series

def gen_custom_exp_series(gamma, T, dt, lam, duration_dist, amplitude_dist, noise_type = None, noise_to_signal_ratio = None):
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    pulse = ps.ExponentialShortPulseGenerator(lam)
    custom_forcing = pm.StandardForcingGenerator()
    custom_forcing.set_duration_distribution(duration_dist)
    custom_forcing.set_amplitude_distribution(amplitude_dist)
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(custom_forcing)

    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
            
    return t, S


    

    
    

def gen_gaussian_series(gamma, T, dt, duration, amp, noise_type = None, noise_to_signal_ratio = None):
    if gamma == 'inf':
        return gaussian_convolution(T, dt, duration) 

    else:   
        series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
        custom_forcing = pm.StandardForcingGenerator()
        pulse = ps.GaussianShortPulseGenerator()
        custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
        custom_forcing.set_amplitude_distribution(lambda k: sst.rayleigh.rvs(scale=np.sqrt(2 / np.pi) * amp, size = k))
        series.set_pulse_shape(pulse)
        series.set_custom_forcing_generator(custom_forcing)
    
        if noise_type is not None:
            series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
            
        t, S = series.make_realization()
                
        return t, S


def gaussian_convolution(T, dt, duration):
    events = np.random.default_rng().standard_normal(int(T/dt))
    cutoff = int((duration * np.sqrt(-2 * np.log(duration * np.sqrt(2 * np.pi) * 1e-50))) / dt )
    pulsedomain = np.arange(-cutoff, cutoff + 1) * dt
    pulse = np.exp(-(pulsedomain / duration)**2 / 2) * 1 / (duration * np.sqrt(2 * np.pi))
    series = ssi.fftconvolve(events, pulse, mode='same')
    t = np.linspace(0, T, int(T / dt))
    
    return t, series

def gen_lorentz_series(gamma, T, dt, duration, noise_type = None, noise_to_signal_ratio = None):
    if gamma == 'inf':
        return lorentz_convolution(T, dt, duration) 

    else:   
        series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
        custom_forcing = pm.StandardForcingGenerator()
        pulse = ps.LorentzShortPulseGenerator(tolerance = 1e-8)
        custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
        #custom_forcing.set_amplitude_distribution(lambda k: 1* np.ones(k))
        series.set_pulse_shape(pulse)
        series.set_custom_forcing_generator(custom_forcing)
    
        if noise_type is not None:
            series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
            
        t, S = series.make_realization()
                
        return t, S
    
def lorentz_convolution(T, dt, duration):
    events = np.random.default_rng().standard_normal(int(T/dt))
    cutoff = int((duration * np.sqrt(-2 * np.log(duration * np.sqrt(2 * np.pi) * 1e-50))) / dt )
    pulsedomain = np.arange(-cutoff, cutoff + 1) * dt
    pulse = np.exp(-(pulsedomain / duration)**2 / 2) * 1 / (duration * np.sqrt(2 * np.pi))
    series = ssi.fftconvolve(events, pulse, mode='same')
    t = np.linspace(0, T, int(T / dt))
    
    return t, series

def trim_transient(times, series, trimsize):
    
    return times[0 : -trimsize * 2], series[trimsize : - trimsize]


def gen_triangle_series(gamma, T, dt, duration, noise_type = None, noise_to_signal_ratio = None): 
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    custom_forcing = pm.StandardForcingGenerator()
    pulse = ps.TriangularShortPulseGenerator()
    custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
    #custom_forcing.set_amplitude_distribution(lambda k: 3 * duration * np.sqrt(2*np.pi) * np.ones(k))
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(custom_forcing)

    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
            
    return t, S

def gen_box_series(gamma, T, dt, duration, noise_type = None, noise_to_signal_ratio = None): 
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    custom_forcing = pm.StandardForcingGenerator()
    pulse = ps.BoxShortPulseGenerator()
    custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
    #custom_forcing.set_amplitude_distribution(lambda k: 3 * duration * np.sqrt(2*np.pi) * np.ones(k))
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(custom_forcing)

    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
            
    return t, S

def gen_gamma_series(gamma, T, dt, alpha, duration, noise_type = None, noise_to_signal_ratio = None): 
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    custom_forcing = pm.StandardForcingGenerator()
    pulse = ps.GammaShortPulseGenerator(alpha)
    custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
    #custom_forcing.set_amplitude_distribution(lambda k: 3 * duration * np.sqrt(2*np.pi) * np.ones(k))
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(custom_forcing)

    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
            
    return t, S

def gen_rayleigh_series(gamma, T, dt, duration, noise_type = None, noise_to_signal_ratio = None): 
    series = pm.PointModel(gamma=gamma, total_duration=T, dt=dt)
    custom_forcing = pm.StandardForcingGenerator()
    pulse = ps.RayleighShortPulseGenerator()
    custom_forcing.set_duration_distribution(lambda k: duration * np.ones(k))
    #custom_forcing.set_amplitude_distribution(lambda k: 3 * duration * np.sqrt(2*np.pi) * np.ones(k))
    series.set_pulse_shape(pulse)
    series.set_custom_forcing_generator(custom_forcing)

    if noise_type is not None:
        series.add_noise(noise_to_signal_ratio = noise_to_signal_ratio, noise_type = noise_type, seed=None)
        
    t, S = series.make_realization()
            
    return t, S

