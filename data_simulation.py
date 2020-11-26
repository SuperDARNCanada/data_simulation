import json
import numpy as np
import math
import itertools
import matplotlib
import pydarnio
import datetime
import copy

matplotlib.use('TkAgg')
from matplotlib import pyplot

C = 299792458

def plotting_acfs(averaging_period_acfs, model, lags, t, amplitude):
    """
    Plots one averaging period of simulated ACFs, along with the mean of those ACFs and true value
    of the model.

    :param      averaging_period_acfs:  The averaging period acfs
    :type       averaging_period_acfs:  ndarray [num_sequences, num_lags]
    :param      model:                  The ACF model used to generate the PDF of which samples
                                        were drawn from.
    :type       model:                  ndarray [all_lags, 1]
    :param      lags:                   The lags defined for the experiment in the config.
    :type       lags:                   ndarray [num_lags, 2]
    :param      t:                      The time steps used to generate each lag of the model ACF.
    :type       t:                      ndarray [all_lags, 1]
    :param      amplitude:              The modelled ACF amplitude.
    :type       amplitude:              int
    """

    lag_nums = lags[:,1] - lags[:,0]
    num_averages = averaging_period_acfs.shape[1]
    model = model[lag_nums]
    t = t[0,lag_nums]

    fig = pyplot.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    tmp=ax.plot(t, averaging_period_acfs.T.real,'grey', alpha=1/np.log10(num_averages)/3)
    tmp=ax.plot(t, model.real, 'k', lw=2)
    tmp=ax.plot(t, np.mean(averaging_period_acfs.T.real, axis=1), '--r')
    tmp=ax.set_ylim([-2 * amplitude, 2 * amplitude])
    tmp=ax.set_title('real')
    tmp=ax.set_xlabel('time (s)')

    ax = fig.add_subplot(122)
    tmp=ax.plot(t, averaging_period_acfs.T.imag, 'grey', alpha=1/np.log10(num_averages)/3)
    tmp=ax.plot(t, model.imag, 'k', lw=2)
    tmp=ax.plot(t, np.mean(averaging_period_acfs.T.imag, axis=1), '--r')
    tmp=ax.set_ylim([-2 * amplitude, 2 * amplitude])
    tmp=ax.set_title('imag')
    tmp=ax.set_xlabel('time (s)')
    pyplot.show()

def plotting_cov_mat(cov_mat):
    """
    Plots the covariance matrix.

    :param      cov_mat:  The covariance matrix.
    :type       cov_mat:  ndarray [2*highest_lag, 2*highest_lag]
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    tmp=ax.imshow(cov_mat)
    pyplot.show()


def create_output_params(transmit_freqs, white_noise_level, sample_separation, pulses, lags,
                        first_range, range_separation, fundamental_lag_spacing, sim_acfs, bfiq_samps):

    num_records, num_ranges, num_averages = sim_acfs.shape[:3]

    now = datetime.datetime.utcnow()
    sequence_time = pulses[-1] * (fundamental_lag_spacing * 1e-6) + (num_ranges * sample_separation * 1e-6)
    sequence_times = [now + datetime.timedelta(seconds=(i * sequence_time)) for i in range(num_records)]
    sequence_times = [dt.timestamp() for dt in sequence_times]

    common_fields = {}
    common_fields['beam_azms'] = np.zeros(num_records, dtype=np.float64)
    common_fields['beam_nums'] = np.zeros(num_records, dtype=np.uint32)
    common_fields['blanked_samples'] = np.zeros(num_records, dtype=np.uint32)
    common_fields['borealis_git_hash'] = np.array(['v0.5'])[0]
    common_fields['data_normalization_factor'] = np.float64(1.0)
    common_fields['experiment_comment'] = np.array(["Simulated data"])[0]
    common_fields['experiment_id']  = np.int64(0)
    common_fields['experiment_name'] = np.array(["data_simulation"])[0]
    common_fields['first_range_rtt'] = np.float32(first_range * sample_separation)
    common_fields['int_time'] = np.ones(num_records, dtype=np.float32) * sequence_time
    common_fields['intf_antenna_count'] = np.uint32(4)
    common_fields['main_antenna_count'] = np.uint32(16)
    common_fields['noise_at_freq'] = np.zeros((num_records, num_averages))
    common_fields['num_sequences'] = np.ones(num_records, dtype=np.int64) * num_averages
    common_fields['num_slices'] = np.ones(num_records, dtype=np.int64) * transmit_freqs.shape[0]
    common_fields['pulses'] = pulses.astype(np.uint32)
    common_fields['range_sep'] = np.float32(range_separation)
    common_fields['rx_sample_rate'] = np.float64(1 / sample_separation)
    common_fields['samples_data_type'] = np.array(['complex float'])[0]
    common_fields['scan_start_marker'] = np.zeros(num_records, dtype=np.bool)
    common_fields['scheduling_mode'] = np.array(['None'])[0]
    common_fields['slice_comment'] = np.array(["Simulated data"])[0]
    common_fields['num_blanked_samples'] = np.zeros(num_records, np.uint32)
    common_fields['num_beams'] = np.ones(num_records, dtype=np.uint32)
    common_fields['first_range'] = np.float32(first_range * range_separation)
    common_fields['slice_interfacing'] = np.array(['']*num_records)
    common_fields['sqn_timestamps'] = np.array(sequence_times)
    common_fields['station'] = np.array(['sas'])[0]
    common_fields['tau_spacing'] = np.uint32(fundamental_lag_spacing)
    common_fields['tx_pulse_len'] = np.uint32(sample_separation)

    shape = list(lags.shape)
    shape[0] += 1

    new_lags = np.zeros(shape)
    new_lags[:-1,:] = lags
    new_lags[-1,:] = pulses[-1]
    common_fields['lags'] = lags.astype(np.uint32)


    for i in range(transmit_freqs.shape[0]):
        rawacf_data = copy.deepcopy(common_fields)
        rawacf_data['averaging_method'] = np.array(['mean'])[0]
        rawacf_data['slice_id'] = np.uint32(i)
        rawacf_data['freq'] = np.uint32(transmit_freqs[i])
        rawacf_data['correlation_descriptors'] = np.array(['num_records', 'num_beams', 'num_ranges', 'num_lags'])
        rawacf_data['main_acfs'] = sim_acfs[:,i].astype(np.complex64)
        rawacf_data['xcfs'] = np.zeros(sim_acfs[:,i].shape, dtype=np.complex64)
        rawacf_data['intf_acfs'] = np.zeros(sim_acfs[:,i].shape, dtype=np.complex64)

        bfiq_data = copy.deepcopy(common_fields)
        bfiq_data['slice_id'] = np.uint32(i)
        bfiq_data['antenna_arrays_order'] = np.array(['main', 'interferometer'])
        bfiq_data['data'] = bfiq_samps[:,i].astype(np.complex64)
        bfiq_data['freq'] = np.uint32(transmit_freqs[i])
        bfiq_data['data_descriptors'] = np.array(['num_records', 'num_antenna_arrays', 'max_num_sequences', 'max_num_beams', 'num_samps'])
        bfiq_data['pulse_phase_offset'] = np.array([], dtype=np.float32)
        bfiq_data['num_samps'] = np.uint32(bfiq_samps.shape[-1])
        bfiq_data['num_ranges'] = np.uint32(num_ranges)



        filename = "simulated_{}.rawacf.hdf5".format(i)
        pydarnio.BorealisWrite(filename, rawacf_data, 'rawacf', 'array')

        filename = "simulated_{}.bfiq.hdf5".format(i)
        pydarnio.BorealisWrite(filename, bfiq_data, 'bfiq', 'array')




def main():

    with open('config.json', 'r') as f:
        sim_params = json.load(f)

    transmit_freqs = np.array(sim_params['transmit_freqs'])
    spectral_width_range = sim_params['spectral_width_range']
    white_noise_level = sim_params['white_noise_level']
    amplitude = sim_params['amplitude']
    sample_separation = sim_params['sample_separation']
    pulses = np.array(sim_params['pulses'])
    num_ranges = sim_params['num_ranges']
    first_range = sim_params['first_range']
    num_averages = sim_params['num_averages']
    range_separation = sim_params['range_separation']
    fundamental_lag_spacing = sim_params['fundamental_lag_spacing']
    velocity_range = sim_params['velocity_range']
    num_records = sim_params['num_records']
    lags = np.array(sim_params['lags'])
    ranges_with_data = sim_params['ranges_with_data']


    # Find highest lag (account for lag 0) and generate all possible lags.
    highest_lag = pulses[-1] + 1
    tmp = np.arange(highest_lag)
    all_possible_lags = np.array(list(itertools.product(tmp, tmp)))


    # Compute the ACF model for all possible lag numbers.
    wavelength = C/(transmit_freqs * 1e3)

    lag_nums = np.abs(all_possible_lags[:,1] - all_possible_lags[:,0])
    t = lag_nums * (fundamental_lag_spacing * 1e-6)
    t = t[np.newaxis,:,np.newaxis]

    v = np.linspace(velocity_range[0], velocity_range[1], num=num_ranges)
    v = v[:,np.newaxis,np.newaxis]

    w = np.linspace(spectral_width_range[0], spectral_width_range[1], num=num_ranges)
    w = w[:,np.newaxis,np.newaxis]

    W_constant = (-1 * 2 * np.pi * t)/wavelength
    V_constant = (1j * 4 * np.pi * t)/wavelength

    acf_model = amplitude * np.exp(W_constant * w) * np.exp(V_constant * v)

    # define rho now to be an array of [4, all_lags]. Each row will be the rho value corresponding
    # to the components of the voltage samples used to make the correlations. The resultant array
    # dimensions can be transposed and reshaped to yield the values of the covariance matrix with
    # dimensions of [2*highest_lag, 2*highest_lag]. Covariance matrix signs will be fixed after.


    rho = np.array([acf_model.real, acf_model.imag, acf_model.imag, acf_model.real])
    rho = np.einsum('ijkl->ljki', rho)

    new_shape1 = [transmit_freqs.shape[0], num_ranges, highest_lag, highest_lag, 2, 2]
    new_axis = (0,1,2,4,3,5)

    new_shape2 = [transmit_freqs.shape[0], num_ranges, highest_lag * 2, highest_lag * 2]

    cov_mat = rho.reshape(new_shape1).transpose(new_axis).reshape(new_shape2)
    cov_mat /= 2.0

    # Signs need to be flipped for every second value on every second diagonal to properly account
    # for the sign changes.
    rows, cols = np.indices((highest_lag * 2, highest_lag * 2))
    signs = np.ones((highest_lag * 2, highest_lag * 2))

    starting_diagonal = (-1 * 2 * highest_lag) + 1
    for i in range(2 * highest_lag):
        diag_to_use = starting_diagonal + (2 * i)

        rows_idx = np.diag(rows, diag_to_use)[::2]
        cols_idx = np.diag(cols, diag_to_use)[::2]

        signs[rows_idx,cols_idx] *= -1.0

    cov_mat *= signs[np.newaxis, np.newaxis, :, :]

    # create the covariance matrix of the noise.
    noise_cov = np.diagflat(np.ones(2 * highest_lag)) * white_noise_level

    np.random.seed(13873) # so that we can deterministically reproduce our results

    # Draw random samples from PDFs generated from the cov_mat.
    size = (num_records, num_averages, 1)
    mean = np.zeros(highest_lag * 2)

    rand_samps = []
    noise_samps = []
    for i in range(transmit_freqs.shape[0]):
        rs = []
        ns = []

        for j in range(num_ranges):
            rs.append(np.random.multivariate_normal(mean, cov_mat[i,j], size=size))
            ns.append(np.random.multivariate_normal(mean, noise_cov, size=size))

        rand_samps.append(rs)
        noise_samps.append(ns)


    rand_samps = np.array(rand_samps)

    ranges_with_data = np.concatenate([np.arange(rng[0],rng[1]) for rng in ranges_with_data])
    mask = np.full(rand_samps.shape, True, dtype=bool)
    mask[:,ranges_with_data,...] = False

    rand_samps[mask] = 0.0

    noise_samps = np.array(noise_samps)

    samps = rand_samps + noise_samps
    samps = np.einsum('ijk...->kij...', samps)


    # Generate ACFs from drawn samples.
    samples_T = samps[...,0::2] + 1j * samps[...,1::2]
    samples_H = np.conj(samples_T)

    samples = np.einsum('...ij->...ji', samples_T)

    all_acfs = np.einsum('...ik,...kj->...ij', samples, samples_H)

    sim_acfs = all_acfs[...,lags[:,0],lags[:,1]]
    sim_acfs = np.mean(sim_acfs, axis=3)

    # add axis for num beams
    sim_acfs = sim_acfs[:,:,np.newaxis,:,:]
    pwr0 = np.abs(sim_acfs)[...,0]

    acfd = np.zeros(sim_acfs.shape + (2,))
    acfd[...,0] = sim_acfs.real
    acfd[...,1] = sim_acfs.imag

    samples_temp = samples[...,pulses,0]
    samples_temp = np.einsum('ijklm->ijlkm', samples_temp)

    num_output_samps = int(first_range + num_ranges + (pulses[-1] * (fundamental_lag_spacing / sample_separation)))
    bfiq_samps = np.zeros(samples_temp.shape[:3] + (num_output_samps,), dtype=samples_temp.dtype)

    # add axis for antenna arrays and for beams.
    bfiq_samps[:,:,np.newaxis,:,np.newaxis,:]

    for i in range(num_ranges):
        idx = pulses * int(fundamental_lag_spacing / sample_separation) + i + first_range
        bfiq_samps[...,idx] += samples_temp[...,i,:]


    create_output_params(transmit_freqs, white_noise_level, sample_separation, pulses, lags,
                        first_range, range_separation, fundamental_lag_spacing, sim_acfs, bfiq_samps)


if __name__ == "__main__":
    main()