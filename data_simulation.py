import json
import numpy as np
import math
import itertools
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot

C = 299792458

with open('config.json', 'r') as f:
    sim_params = json.load(f)

transmit_freq = sim_params['transmit_freq']
spectral_width = sim_params['spectral_width']
white_noise_level = sim_params['white_noise_level']
amplitude = sim_params['amplitude']
sample_separation = sim_params['sample_separation']
pulses = np.array(sim_params['pulses'])
num_ranges = sim_params['num_ranges']
first_range = sim_params['first_range']
num_averages = sim_params['num_averages']
range_separation = sim_params['range_separation']
fundamental_lag_spacing = sim_params['fundamental_lag_spacing']
velocity = sim_params['velocity']
num_records = sim_params['num_records']
lags = np.array(sim_params['lags'])


highest_lag = pulses[-1] + 1
tmp = np.arange(highest_lag)
all_possible_lags = np.array(list(itertools.product(tmp, tmp)))


wavelength = C/(transmit_freq * 1e3)

lag_nums = np.abs(all_possible_lags[:,1] - all_possible_lags[:,0])
t = lag_nums * (fundamental_lag_spacing * 1e-6)


W_constant = (-1 * 2 * np.pi * t)/wavelength
V_constant = (1j * 4 * np.pi * t)/wavelength

acf_model = amplitude * np.exp(W_constant * spectral_width) * np.exp(V_constant * velocity)

rho = np.array([acf_model.real, acf_model.imag, acf_model.imag, acf_model.real])

new_shape1 = [highest_lag, highest_lag, 2, 2]
new_axis = (0,2,1,3)

new_shape2 = [highest_lag*2, highest_lag*2]

cov_mat = rho.T.reshape(new_shape1).transpose(new_axis).reshape(new_shape2)
cov_mat /= 2.0


rows, cols = np.indices(cov_mat.shape)

starting_diagonal = (-1 * 2 * highest_lag) + 1
for i in range(2 * highest_lag):
    diag_to_use = starting_diagonal + (2 * i)

    rows_idx = np.diag(rows, diag_to_use)[::2]
    cols_idx = np.diag(cols, diag_to_use)[::2]

    cov_mat[rows_idx,cols_idx] *= -1.0


# tst_cov = amplitude * np.diagflat(np.ones(2*highest_lag))
# for j in range(0,2*highest_lag,2):
#     for k in range(j+2,2*highest_lag,2):
#         rho = acf_model[(k-j)//2]
#         #upper diagonal blocks
#         tst_cov[j,k] = rho.real
#         tst_cov[j,k+1] = -1 * rho.imag
#         tst_cov[j+1,k] = rho.imag
#         tst_cov[j+1,k+1] = rho.real
#         #lower diagonal blocks
#         tst_cov[k,j] = rho.real
#         tst_cov[k,j+1] = rho.imag
#         tst_cov[k+1,j] =  -1 * rho.imag
#         tst_cov[k+1,j+1] = rho.real

# tst_cov /= 2

noise_cov = np.diagflat(np.ones(2 * highest_lag)) * white_noise_level

np.random.seed(13873) # so that we can deterministically reproduce our results


rand_samps = np.random.multivariate_normal(np.zeros(highest_lag*2), cov_mat, size=(num_averages,1))
noise_samps = np.random.multivariate_normal(np.zeros(highest_lag*2), noise_cov, size=(num_averages,1))

samps = rand_samps + noise_samps

samples_T = samps[...,0::2] + 1j * samps[...,1::2]
samples_H = np.conj(samples_T)

samples = np.einsum('...ij->...ji', samples_T)

all_acfs = np.einsum('...ik,...kj->...ij', samples, samples_H)

sim_acfs = all_acfs[...,lags[:,0],lags[:,1]]

tmp = np.zeros(samples.shape[0:2],dtype=complex) * (np.nan + 1j*np.nan)
lag_nums = lags[:,1] - lags[:,0]

tmp[:,lag_nums] = sim_acfs
sim_acfs = tmp.T



ll = all_possible_lags[:pulses[-1]+1, :]

lag_nums = np.abs(ll[:,1] - ll[:,0])
t = lag_nums * (fundamental_lag_spacing * 1e-6)

W_constant = (-1 * 2 * np.pi * t)/wavelength
V_constant = (1j * 4 * np.pi * t)/wavelength

acf_model = amplitude * np.exp(W_constant * spectral_width) * np.exp(V_constant * velocity)


fig = pyplot.figure(figsize=(12,6))
ax = fig.add_subplot(121)
tmp=ax.plot(t,sim_acfs.real,'grey',alpha=1/np.log10(num_averages)/3)
tmp=ax.plot(t,acf_model.real,'k',lw=2)
tmp=ax.plot(t,np.mean(sim_acfs.real,axis=1),'--r')
tmp=ax.set_ylim([-2*amplitude,2*amplitude])
tmp=ax.set_title('real')
tmp=ax.set_xlabel('time (s)')

ax = fig.add_subplot(122)
tmp=ax.plot(t,sim_acfs.imag,'grey',alpha=1/np.log10(num_averages)/3)
tmp=ax.plot(t,acf_model.imag,'k',lw=2)
tmp=ax.plot(t,np.mean(sim_acfs.imag,axis=1),'--r')
tmp=ax.set_ylim([-2*amplitude,2*amplitude])
tmp=ax.set_title('imag')
tmp=ax.set_xlabel('time (s)')
pyplot.show()

fig = pyplot.figure()
ax = fig.add_subplot(111)
tmp=ax.imshow(cov_mat)
pyplot.show()
