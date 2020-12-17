import pydarnio
import datetime
import math
import json
import numpy as np
import deepdish as dd

fitacf_file = "20201201.0001.00.sas.0.fitacf"
bfiq_file = "20201201.0001.00.sas.0.bfiq.hdf5"

outname = "data_sim_test_set.hdf5"

record_num = 104

fitacf_records = pydarnio.SDarnRead(fitacf_file).read_fitacf()
bfiq_records = pydarnio.BorealisRead(bfiq_file, 'bfiq', 'array').arrays

fitacf_record = fitacf_records[record_num]

params = {}

p_l_abs_units = fitacf_record['noise.sky'] * (10 ** (fitacf_record['p_l']/10.0))
p_l_abs_units /= ((np.iinfo(np.int16).max ** 2) / (bfiq_records['data_normalization_factor'] ** 2))

amplitudes = np.ones(fitacf_record['nrang'])
amplitudes[fitacf_record['slist']] = p_l_abs_units
params['amplitudes'] = amplitudes

spectral_widths = np.ones(fitacf_record['nrang'])
spectral_widths[fitacf_record['slist']] = fitacf_record['w_l']
params['spectral_widths'] = spectral_widths

velocities = np.ones(fitacf_record['nrang'])
velocities[fitacf_record['slist']] = fitacf_record['v']
params['velocities'] = velocities

elevation_phases = np.ones(fitacf_record['nrang'])
elevation_phases[fitacf_record['slist']] = fitacf_record['phi0']
params['elevation_phases'] = elevation_phases

params['num_ranges'] = fitacf_record['nrang']
params['num_averages'] = fitacf_record['nave']
params['range_separation'] = fitacf_record['rsep']
params['pulses'] = fitacf_record['ptab']
params['lags'] = fitacf_record['ltab']
params['fundamental_lag_spacing'] = fitacf_record['mpinc']
params['sample_separation'] = fitacf_record['txpl']
params['transmit_freqs'] = np.array([fitacf_record['tfreq']])
params['noise_level'] = fitacf_record['noise.sky']
params['ranges_with_data'] = fitacf_record['slist']
params['first_range'] = int(fitacf_record['frang']/fitacf_record['rsep'])
params['seed'] = 1357911

params['timestamp'] = bfiq_records['sqn_timestamps'][record_num,0]
params['fitacf_file'] = fitacf_file
params['bfiq_file'] = bfiq_file
params['iq_data'] = bfiq_records['data'][record_num]
params['blanked_samples'] = bfiq_records['blanked_samples'][record_num]

dd.io.save(outname, params)