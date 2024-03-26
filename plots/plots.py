import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (AnchoredOffsetbox, HPacker, TextArea)
from lampe.plots import nice_rc, corner
from astropy.time import Time
from orbitize.kepler import calc_orbit

def plot_orbit(theta, data, label="Simulated data", timestep = None, text = False):
    """
    Plots the orbit based on the given theta values and data.

    Parameters:
    theta (numpy.ndarray): Array of parameter values.
    data (numpy.ndarray): Array of data points.
    """
    
    ra = data[0::2]
    dec = data[1::2]

    label_print = ['a', 'e', 'i', 'ω', 'Ω', 'τ', 'Mt']

    # Just plot the values different from 0
    non_zero_indices = np.where((ra != 0) & (dec != 0))
    ra = ra[non_zero_indices]
    dec = dec[non_zero_indices]

    if timestep is None:
        color = np.linspace(2002, 2019.9, 180)[non_zero_indices]
        print("Color is " + str(color))
    else:
        color = timestep

    fig, ax = plt.subplots() 

    shw = ax.scatter(ra, dec, c=color, cmap='winter', label=label)
    bar = plt.colorbar(shw)

    plt.xlabel('$\\Delta$ RA'); plt.ylabel('$\\Delta$ Dec')
    plt.axis('equal')
    plt.plot(0,0,marker="*",color='black',markersize=10, label = "Star", linestyle='')
    bar.set_label('Year')
    plt.legend()
    
    if text:
        box1 = TextArea('\n'.join([f"{label}: {theta_value:.3f}" for label, theta_value in zip(label_print[0::4], theta.tolist()[0::4])]))
        box2 = TextArea('\n'.join([f"{label}: {theta_value:.3f}" for label, theta_value in zip(label_print[1::4], theta.tolist()[1::4])]))
        box3 = TextArea('\n'.join([f"{label}: {theta_value:.3f}" for label, theta_value in zip(label_print[2::4], theta.tolist()[2::4])]))
        box4 = TextArea('\n'.join([f"{label}: {theta_value:.3f}\n" for label, theta_value in zip(label_print[3::4], theta.tolist()[3::4])]))
        box_title = TextArea("Parameters : \n")
        
        box = HPacker(children=[box_title, box1, box2, box3, box4],
                    align="left",
                    pad=0, sep=5) 
        
        anchored_box = AnchoredOffsetbox(loc='lower left',
                                        child=box, pad=0.5,
                                        frameon=True,
                                        bbox_to_anchor=(0., 1.02),
                                        bbox_transform=ax.transAxes,
                                        borderpad=0.,)
        ax.add_artist(anchored_box)
        fig.subplots_adjust(top=0.8)
    plt.show()

def corner_plot(npe_samples, mcmc_samples):
    LOWER = torch.tensor([8.0, 0.0, 85.5, 0.0, 29.6, 0.0, 50.8, 1.5])
    UPPER = torch.tensor([40.0, 0.9, 93.0, 360.0, 33.0, 1.0, 52.1, 2.0])

    LABELS = [r'$a$', r'$e$', r'$i$',
            r'$\omega$', r'$\Omega$',
            r'$\tau$', r'$\pi$', r'$M_T$']

    plt.rcParams.update(nice_rc(latex=True))
    fig = corner(
        npe_samples,
        domain=(LOWER, UPPER),
        bins=64,
        #smooth=2,
        labels=LABELS,
        legend=r'$p_\phi(\theta | x^*)$',
    )

    corner(
        mcmc_samples,
        domain=(LOWER, UPPER),
        bins=64,
        #smooth=2,
        labels=LABELS,
        legend=r'MCMC',
        figure = fig
    )

    plt.show()

def radec_plot(samples, data_table):
    if type(samples) == torch.Tensor:
        sma, ecc, inc, aop, pan, tau, plx, mtot = samples[0:5000].T.numpy()
    else:
        sma, ecc, inc, aop, pan, tau, plx, mtot = samples[0:5000].T

    inc = np.radians(inc)
    aop = np.radians(aop)
    pan = np.radians(pan)

    t_obs = np.linspace(2002, 2019.9, 180)
    t = np.linspace(2000, 2029.9, 180)
    observation_epochs = Time(t, format='decimalyear').mjd

    ra, dec, _ = calc_orbit(observation_epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, use_c = False, tau_ref_epoch=50000)
    ra = ra.T
    dec = dec.T

    fig, axs = plt.subplots(2)

    axs[0].fill_between(t,
                    np.quantile(ra, 0.0015, axis=0),
                    np.quantile(ra, 0.9985, axis=0),
                    color = "#1f77b4",
                    alpha=0.3,
                    label = "99.7\%")
    axs[0].fill_between(t,
                    np.quantile(ra, 0.0225, axis=0),
                    np.quantile(ra, 0.9775, axis=0),
                    color = "#1f77b4",
                    alpha=0.4,
                    label = "95.5\%")
    axs[0].fill_between(t,
                    np.quantile(ra, 0.1585, axis=0),
                    np.quantile(ra, 0.8415, axis=0),
                    color = "#1f77b4",
                    alpha=0.5,
                    label = "68.3\%")

    axs[1].fill_between(t,
                    np.quantile(dec, 0.0015, axis=0),
                    np.quantile(dec, 0.9985, axis=0),
                    color = "#1f77b4",
                    alpha=0.3,
                    label = "99.7\%")
    axs[1].fill_between(t,
                    np.quantile(dec, 0.0225, axis=0),
                    np.quantile(dec, 0.9775, axis=0),
                    color = "#1f77b4",
                    alpha=0.4,
                    label = "95.5\%")
    axs[1].fill_between(t,
                    np.quantile(dec, 0.1585, axis=0),
                    np.quantile(dec, 0.8415, axis=0),
                    color = "#1f77b4",
                    alpha=0.5,
                    label = "68.3\%")


    axs[0].errorbar(Time(data_table['epoch'], format='mjd').decimalyear, 
                    data_table["quant1"], 
                    yerr = data_table["quant1_err"],
                    fmt="o", 
                    markersize=0.5,
                    color = "r", 
                    zorder=1, 
                    label = "Observations")
    axs[0].set_ylabel('$\\Delta$ RA')
    axs[0].set_ylim(-300, 500)
    axs[0].set_xlim(2000, 2025)
    axs[0].grid()
    axs[0].legend(bbox_to_anchor=(1.05, 1),
                            loc='upper left', borderaxespad=0., title = "Confidence interval")

    axs[1].errorbar(Time(data_table['epoch'], format='mjd').decimalyear, 
                    data_table["quant2"], 
                    yerr = data_table["quant2_err"],
                    fmt="o", 
                    markersize=0.5,
                    color = "r", 
                    zorder=1, 
                    label = "Observations")
    axs[1].set_ylabel('$\\Delta$ DEC')
    axs[1].set_ylim(-500, 1000)
    axs[1].set_xlim(2000, 2025)
    axs[1].grid()

    axs[1].set_xlabel('Time')

    plt.show()