import json
import numpy as np
import math
import itertools
import matplotlib
import pydarnio
import datetime

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
    t = t[lag_nums]

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


def create_output_params(transmit_freq, white_noise_level, sample_separation, pulses, lags,
                        first_range, range_separation, fundamental_lag_spacing, sim_acfs):

    num_records, num_ranges, num_averages = sim_acfs.shape[:3]

    now = datetime.datetime.utcnow()
    sequence_time = pulses[-1] * (fundamental_lag_spacing * 1e-6) + (num_ranges * sample_separation * 1e-6)
    sequence_times = [now + datetime.timedelta(seconds=(i * sequence_time)) for i in range(num_records)]

    pwr0 = np.abs(sim_acfs)[...,0]

    acfd = np.zeros(sim_acfs.shape + (2,))
    acfd[...,0] = sim_acfs.real
    acfd[...,1] = sim_acfs.imag

    shape = list(lags.shape)
    shape[0] += 1

    new_lags = np.zeros(shape)
    new_lags[:-1,:] = lags
    new_lags[-1,:] = pulses[-1]

    records = []
    for i in range(num_records):
        record = {}
        record['radar.revision.major' ] = np.int8(0)
        record['radar.revision.minor'] = np.int8(0)
        record['origin.code'] = np.int8(0)
        record['origin.time'] = ' '
        record['origin.command'] = 'python3 data_simulation.py'
        record['cp'] = np.int16(0)
        record['stid'] = np.int16(6)
        record['time.yr'] = np.int16(sequence_times[i].year)
        record['time.mo'] = np.int16(sequence_times[i].month)
        record['time.dy'] = np.int16(sequence_times[i].day)
        record['time.hr'] = np.int16(sequence_times[i].hour)
        record['time.mt'] = np.int16(sequence_times[i].minute)
        record['time.sc'] = np.int16(sequence_times[i].second)
        record['time.us'] = np.int32(sequence_times[i].microsecond)
        record['txpow'] = np.int16(0)
        record['nave'] = np.int16(num_averages)
        record['atten'] = np.int16(0)
        record['lagfr'] = np.int16(first_range * sample_separation)
        record['smsep'] = np.int16(sample_separation)
        record['ercod'] = np.int16(0)
        record['stat.agc'] = np.int16(0)
        record['stat.lopwr'] = np.int16(0)
        record['noise.search'] = np.float32(0)
        record['noise.mean'] = np.float32(0)
        record['channel'] = np.int16(0)
        record['bmnum'] = np.int16(0)
        record['bmazm'] = np.float32(0)
        record['scan'] = np.int16(0)
        record['offset'] = np.int16(0)
        record['rxrise'] = np.int16(0)

        integer, decimal = divmod(sequence_time * num_averages, 1)
        record['intt.sc'] = np.int16(integer)
        record['intt.us'] = np.int32(decimal)
        record['txpl'] = np.int16(sample_separation)
        record['mpinc'] = np.int16(fundamental_lag_spacing)
        record['mppul'] = np.int16(pulses.shape[0])
        record['mplgs'] = np.int16(lags.shape[0])
        record['nrang'] = np.int16(num_ranges)
        record['frang'] = np.int16(first_range * range_separation)
        record['rsep'] = np.int16(range_separation)
        record['xcf'] = np.int16(0)
        record['tfreq'] = np.int16(transmit_freq)
        record['mxpwr'] = np.int32(0)
        record['lvmax'] = np.int32(0)
        record['rawacf.revision.major'] = np.int32(0)
        record['rawacf.revision.minor'] = np.int32(0)
        record['combf'] = "Simulated data"
        record['thr'] = np.float32(0)
        record['ptab'] = pulses.astype(np.int16)
        record['ltab'] = new_lags.astype(np.int16)
        record['slist'] = np.arange(num_ranges, dtype=np.int16)
        record['pwr0'] = pwr0[i].astype(np.float32)
        record['acfd'] = acfd[i].astype(np.float32)

        records.append(record)

    filename = "sim.rawacf"
    writer = pydarnio.SDarnWrite(records)
    writer.write_rawacf(filename)








def main():

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


    # Find highest lag (account for lag 0) and generate all possible lags.
    highest_lag = pulses[-1] + 1
    tmp = np.arange(highest_lag)
    all_possible_lags = np.array(list(itertools.product(tmp, tmp)))


    # Compute the ACF model for all possible lag numbers.
    wavelength = C/(transmit_freq * 1e3)

    lag_nums = np.abs(all_possible_lags[:,1] - all_possible_lags[:,0])
    t = lag_nums * (fundamental_lag_spacing * 1e-6)

    W_constant = (-1 * 2 * np.pi * t)/wavelength
    V_constant = (1j * 4 * np.pi * t)/wavelength

    acf_model = amplitude * np.exp(W_constant * spectral_width) * np.exp(V_constant * velocity)

    # define rho now to be an array of [4, all_lags]. Each row will be the rho value corresponding
    # to the components of the voltage samples used to make the correlations. The resultant array
    # dimensions can be transposed and reshaped to yield the values of the covariance matrix with
    # dimensions of [2*highest_lag, 2*highest_lag]. Covariance matrix signs will be fixed after.

    rho = np.array([acf_model.real, acf_model.imag, acf_model.imag, acf_model.real])

    new_shape1 = [highest_lag, highest_lag, 2, 2]
    new_axis = (0,2,1,3)

    new_shape2 = [highest_lag * 2, highest_lag * 2]

    cov_mat = rho.T.reshape(new_shape1).transpose(new_axis).reshape(new_shape2)
    cov_mat /= 2.0


    # Signs need to be flipped for every second value on every second diagonal to properly account
    # for the sign changes.
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

    # create the covariance matrix of the noise.
    noise_cov = np.diagflat(np.ones(2 * highest_lag)) * white_noise_level

    np.random.seed(13873) # so that we can deterministically reproduce our results

    # Draw random samples from PDFs generated from the cov_mat.
    size = (num_records, num_ranges, num_averages, 1)
    mean = np.zeros(highest_lag * 2)
    rand_samps = np.random.multivariate_normal(mean, cov_mat, size=size)
    noise_samps = np.random.multivariate_normal(mean, noise_cov, size=size)

    samps = rand_samps + noise_samps

    # Generate ACFs from drawn samples.
    samples_T = samps[...,0::2] + 1j * samps[...,1::2]
    samples_H = np.conj(samples_T)

    samples = np.einsum('...ij->...ji', samples_T)

    all_acfs = np.einsum('...ik,...kj->...ij', samples, samples_H)

    sim_acfs = all_acfs[...,lags[:,0],lags[:,1]]
    sim_acfs = np.mean(sim_acfs, axis=2)

    create_output_params(transmit_freq, white_noise_level, sample_separation, pulses, lags,
                        first_range, range_separation, fundamental_lag_spacing, sim_acfs)

    #plotting_acfs(sim_acfs[0,0], acf_model, lags, t, amplitude)
    #plotting_cov_mat(cov_mat)


if __name__ == "__main__":
    main()