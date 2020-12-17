import deepdish as dd
import data_simulation
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

test_set = dd.io.load("data_sim_test_set.hdf5")

np.random.seed(test_set["seed"])

keys = ["amplitudes", "bfiq_file", "elevation_phases", "first_range", "transmit_freqs",
"lags", "noise_level", "num_averages", "num_ranges", "pulses",
"range_separation", "ranges_with_data", "sample_separation", "spectral_widths",
"fundamental_lag_spacing", "velocities"]

sim_params = {}
for k in keys:
    sim_params[k] = test_set[k]

sim_params['num_records'] = 10

iq = []
acfs = []
for _ in range(10):
    sim_data = data_simulation.simulate(sim_params)
    iq.append(sim_data['bfiq'][:,0,0,:,0])
    acfs.append(sim_data['main_acfs'][:,0,0,:,:])

iq = np.array(iq)
iq = iq.reshape((-1, iq.shape[-1]))

acfs = np.array(acfs)
acfs = acfs.reshape((100,) + acfs.shape[2:])

test_set['iq_data'][...,test_set['blanked_samples']] = 0.0 + 0.0j

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,1, hspace=0.5)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.set_title("Verification that simulator can simulate real data")
ax1.plot(iq.T.real, 'k-', lw=1, alpha=0.1, zorder=20)
ax1.plot(test_set['iq_data'][0,:,0].T.real, lw=1, zorder=30)
ax1.plot(np.array([test_set['blanked_samples'],test_set['blanked_samples']]),[-2e5,2e5],"r",lw=1, zorder=40)
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Sample Number")
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax2.hist(iq[:,31].real, bins=50, density=True, color='grey')
ax2.hist(test_set['iq_data'][0,:,0,31].real, bins=10, density=True, histtype='step', color='r')
ax2.set_ylabel("Occurence")
ax2.set_xlabel("Amplitude")
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.savefig("data_simulator_verification_plot.png")


test_outputs = {'simulated_iq' : iq, 'simulated_acfs' : acfs}
dd.io.save('data_simulator_test_outputs.hdf5', test_outputs)



