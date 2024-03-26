import numpy as np
import orbitize
from orbitize import read_input, system, priors, sampler
import matplotlib.pyplot as plt

def main():
    data_table = read_input.read_file('{}/betaPic.csv'.format(orbitize.DATADIR))
    data_table = data_table[:-1] # Discard the RV observation, as we do not take it into account in the model

    num_planets = 1
    total_mass = 1.75 # [Msol]
    plx = 51.44 # [mas]
    mass_err = 0.05 # [Msol]
    plx_err = 0.12 # [mas]

    sys = system.System(
        num_planets, data_table, total_mass,
        plx, mass_err=mass_err, plx_err=plx_err
    )

    lab = sys.param_idx

    # set up the same priors as for https://arxiv.org/abs/2201.08506v1(github : https://github.com/HeSunPU/DPI/tree/main)
    sys.sys_priors[lab['sma1']] = priors.UniformPrior(4.0, 40.0)
    sys.sys_priors[lab['ecc1']] = priors.UniformPrior(0.00001, 0.99)
    sys.sys_priors[lab['inc1']] = priors.UniformPrior(np.deg2rad(81), np.deg2rad(99))
    sys.sys_priors[lab['aop1']] = priors.UniformPrior(0, 2*np.pi)
    sys.sys_priors[lab['pan1']] = priors.UniformPrior(np.deg2rad(25), np.deg2rad(85))
    sys.sys_priors[lab['tau1']] = priors.UniformPrior(0, 1)

    # number of temperatures & walkers for MCMC
    num_temps = 20
    num_walkers = 1000

    # number of steps to take
    n_orbs = 10000000

    mcmc_sampler = sampler.MCMC(sys, num_temps, num_walkers)

    _ = mcmc_sampler.run_sampler(n_orbs, output_filename='mcmc_betapic.hdf5')

if __name__ == '__main__':
    main()