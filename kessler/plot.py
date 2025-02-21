import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import uuid
import tempfile
import pyprob
from pyprob.distributions import Empirical
import numpy as np
import torch

from . import util
from dsgp4 import tle

mpl.rcParams['axes.unicode_minus'] = False

# I need to re-write this w.r.t. pyprob, since 'nonposy', 'nonposx' are deprecated in favour of 'nonpositive'
# TODO: transform this into a more generic plot_priors, that takes the priors dict, and plots each mixture
def plot_mix(mix, min_val=-10, max_val=10, resolution=1000, figsize=(10, 5), xlabel=None, ylabel='Probability', xticks=None, yticks=None, log_xscale=False, log_yscale=False, file_name=None, show=True, fig=None, ax = None, *args, **kwargs):
    if ax is None:
        if not show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
        ax.grid()
    xvals = np.linspace(min_val, max_val, resolution)
    ax.plot(xvals, [torch.exp(mix.log_prob(x)) for x in xvals], *args, **kwargs)
    if log_xscale:
        ax.set_xscale('log')
    if log_yscale:
        ax.set_yscale('log', nonpositive='clip')
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_xticks(yticks)
    # if xlabel is None:
    #     xlabel = mix.name
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if file_name is not None:
        plt.savefig(file_name)
    if show:
        plt.show()
    return ax

def plot_tles(tles, file_name=None, figsize = (36,18), show=True, axs=None, return_axs=False, log_yscale=False, *args, **kwargs):
    """
    This function takes a list of tles as input and plots the histograms of some of their elements.

    Inputs
    ----------
    - tles (`list`): list of tles, where each element is a dictionary.
    - save_fig (`bool`): boolean variable, if True, figure is saved to a file.
    - file_name (`str`): name of the file (including path) where the plot is to be saved.
    - figsize (`tuple`): figure size.

    Outputs
    ----------
    - ax (`numpy.ndarray`): array of AxesSubplot objects
    """
    #I collect all the six variables from the TLEs:
    mean_motion, eccentricity, inclination, argument_of_perigee, raan, b_star, mean_anomaly, mean_motion_first_derivative, mean_motion_second_derivative = tle.tle_elements(tles = tles)

    plt.rcParams.update({'font.size': 22})
    if axs is None:
        fig, axs = plt.subplots(3, 3, figsize = figsize)

    axs[0,0].hist(mean_motion, bins = 100, *args, **kwargs)
    axs[0,0].set_xlabel('Mean Motion [rad/s]')
    x_min, x_max = min(mean_motion), max(mean_motion)
    axs[0,0].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[0,0].set_yscale('log')
    axs[0,0].grid(True)

    axs[0,1].hist(eccentricity, bins = 100, *args, **kwargs)
    axs[0,1].set_xlabel('Eccentricity [-]')
    x_min, x_max = min(eccentricity), max(eccentricity)
    axs[0,1].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[0,1].set_yscale('log')
    axs[0,1].grid(True)

    axs[0,2].hist([i*180/np.pi for i in inclination], bins = 100, *args, **kwargs)
    axs[0,2].set_xlabel('Inclination [deg]')
    x_min, x_max = min(inclination)*180/np.pi, max(inclination)*180/np.pi
    axs[0,2].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[0,2].set_yscale('log')
    axs[0,2].grid(True)

    axs[1,0].hist([omega*180/np.pi for omega in argument_of_perigee], bins = 100, *args, **kwargs)
    axs[1,0].set_xlabel('Argument of Perigee [deg]')
    x_min, x_max = min(argument_of_perigee)*180/np.pi, max(argument_of_perigee)*180/np.pi
    axs[1,0].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[1,0].set_yscale('log')
    axs[1,0].grid(True)

    axs[1,1].hist([RAAN*180/np.pi for RAAN in raan], bins = 100, *args, **kwargs)
    axs[1,1].set_xlabel('RAAN [deg]')
    x_min, x_max = min(raan)*180/np.pi, max(raan)*180/np.pi
    axs[1,1].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[1,1].set_yscale('log')
    axs[1,1].grid(True)

    axs[1,2].hist(b_star, bins = 100, *args, **kwargs)
    axs[1,2].set_xlabel('Bstar [1/m]')
    x_min, x_max = min(b_star), max(b_star)
    axs[1,2].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[1,2].set_yscale('log')
    axs[1,2].grid(True)

    axs[2,0].hist([M*180/np.pi for M in mean_anomaly], bins = 100, *args, **kwargs)
    axs[2,0].set_xlabel('Mean Anomaly [deg]')
    x_min, x_max = min(mean_anomaly)*180/np.pi, max(mean_anomaly)*180/np.pi
    axs[2,0].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[2,0].set_yscale('log')
    axs[2,0].grid(True)

    axs[2,1].hist(mean_motion_first_derivative, bins = 100, *args, **kwargs)
    axs[2,1].set_xlabel('Mean Motion 1st Der [rad/s**2]')
    x_min, x_max = min(mean_motion_first_derivative), max(mean_motion_first_derivative)
    axs[2,1].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[2,1].set_yscale('log')
    axs[2,1].grid(True)

    axs[2,2].hist(mean_motion_second_derivative, bins = 100, *args, **kwargs)
    axs[2,2].set_xlabel('Mean Motion 2nd Der [rad/s**3]')
    x_min, x_max = min(mean_motion_second_derivative), max(mean_motion_second_derivative)
    axs[2,2].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[2,2].set_yscale('log')
    axs[2,2].grid(True)

    if file_name is not None:
        fig.savefig(fname = file_name)
    if show and not return_axs:
        plt.show()

    if return_axs:
        return axs


def plot_dist(dists, file_name=None, n_bins=30, num_resample=None, trace=None, figsize = (16, 18)):
    if isinstance(dists, Empirical):
        dists = [dists]

    marginal_dists = [{} for _ in range(len(dists))]
    pyprob.set_verbosity(0)
    for i, dist in enumerate(dists):
        if num_resample is not None:
            dist = dist.resample(num_resample)
        dist = dist.condition(lambda t: not t['prop_error'])

        marginal_dists[i]['dist_time_min'] = dist.map(lambda t:t['time_min'])
        marginal_dists[i]['dist_d_min'] = dist.map(lambda t:t['d_min'])
        marginal_dists[i]['dist_conj'] = dist.map(lambda t:t['conj'])
        marginal_dists[i]['dist_events_with_conjunction'] = dist.condition(lambda t:t['conj'])
        marginal_dists[i]['dist_time_conj'] = marginal_dists[i]['dist_events_with_conjunction'].map(lambda t:t['time_conj'])
        marginal_dists[i]['dist_d_conj'] = marginal_dists[i]['dist_events_with_conjunction'].map(lambda t:t['d_conj'])

        marginal_dists[i]['dist_t_mean_motion'] = dist.map(lambda t:t['t_mean_motion'])
        marginal_dists[i]['dist_t_mean_anomaly'] = dist.map(lambda t:t['t_mean_anomaly'])
        marginal_dists[i]['dist_t_eccentricity'] = dist.map(lambda t:t['t_eccentricity'])
        marginal_dists[i]['dist_t_inclination'] = dist.map(lambda t:t['t_inclination'])
        marginal_dists[i]['dist_t_argument_of_perigee'] = dist.map(lambda t:t['t_argument_of_perigee'])
        marginal_dists[i]['dist_t_raan'] = dist.map(lambda t:t['t_raan'])
        marginal_dists[i]['dist_t_mean_motion_first_derivative'] = dist.map(lambda t:t['t_mean_motion_first_derivative'])
        marginal_dists[i]['dist_t_b_star'] = dist.map(lambda t:t['t_b_star'])

        marginal_dists[i]['dist_c_mean_motion'] = dist.map(lambda t:t['c_mean_motion'])
        marginal_dists[i]['dist_c_mean_anomaly'] = dist.map(lambda t:t['c_mean_anomaly'])
        marginal_dists[i]['dist_c_eccentricity'] = dist.map(lambda t:t['c_eccentricity'])
        marginal_dists[i]['dist_c_inclination'] = dist.map(lambda t:t['c_inclination'])
        marginal_dists[i]['dist_c_argument_of_perigee'] = dist.map(lambda t:t['c_argument_of_perigee'])
        marginal_dists[i]['dist_c_raan'] = dist.map(lambda t:t['c_raan'])
        marginal_dists[i]['dist_c_mean_motion_first_derivative'] = dist.map(lambda t:t['c_mean_motion_first_derivative'])
        marginal_dists[i]['dist_c_b_star'] = dist.map(lambda t:t['c_b_star'])

        marginal_dists[i]['dist_num_cdms'] = marginal_dists[i]['dist_events_with_conjunction'].map(lambda t:t['num_cdms'])
        if len(marginal_dists[i]['dist_conj']) > 0:
            marginal_dists[i]['dist_time_cdm'] = marginal_dists[i]['dist_events_with_conjunction'].map(lambda t:t['time_cdm'])

    pyprob.set_verbosity(2)

    fig, axs = plt.subplots(8, 4, figsize=figsize)

    t_color = 'green'
    c_color = 'red'
    # Chaser and target
    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[0,0].hist(marginal_dists[i]['dist_t_mean_motion'].values_numpy(), bins=n_bins, alpha=0.5, label=label, density=True)
    # axs[0,0].legend()
    axs[0,0].set_xlabel('mean_motion')
    axs[0,0].set_ylabel('Target')
    if trace:
        axs[0,0].vlines(trace['t_mean_motion'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[0,1].hist(marginal_dists[i]['dist_t_mean_anomaly'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[0,1].legend()
    axs[0,1].set_xlabel('mean_anomaly')
    if trace:
        axs[0,1].vlines(trace['t_mean_anomaly'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[0,2].hist(marginal_dists[i]['dist_t_eccentricity'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[0,2].legend()
    axs[0,2].set_xlabel('eccentricity')
    if trace:
        axs[0,2].vlines(trace['t_eccentricity'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[0,3].hist(marginal_dists[i]['dist_t_inclination'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[0,3].legend()
    axs[0,3].set_xlabel('inclination')
    if trace:
        axs[0,3].vlines(trace['t_inclination'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[1,0].hist(marginal_dists[i]['dist_t_argument_of_perigee'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[1,0].legend()
    axs[1,0].set_xlabel('argument_of_perigee')
    axs[1,0].set_ylabel('Target')
    if trace:
        axs[1,0].vlines(trace['t_argument_of_perigee'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[1,1].hist(marginal_dists[i]['dist_t_raan'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[1,1].legend()
    axs[1,1].set_xlabel('raan')
    if trace:
        axs[1,1].vlines(trace['t_raan'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[1,2].hist(marginal_dists[i]['dist_t_mean_motion_first_derivative'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[1,2].legend()
    axs[1,2].set_xlabel('mean_motion_first_derivative')
    if trace:
        axs[1,2].vlines(trace['t_mean_motion_first_derivative'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, bins, _ = axs[1,3].hist(marginal_dists[i]['dist_t_b_star'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[1,3].legend()
    axs[1,3].set_xlabel('b_star')
    if trace:
        axs[1,3].vlines(trace['t_b_star'], 0, np.max(h)*1.05, linestyles='dashed')
#     ax.set_xlim(-0.01,0.01)



    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[2,0].hist(marginal_dists[i]['dist_c_mean_motion'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[2,0].legend()
    axs[2,0].set_xlabel('mean_motion')
    axs[2,0].set_ylabel('Chaser')
    if trace:
        axs[2,0].vlines(trace['c_mean_motion'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[2,1].hist(marginal_dists[i]['dist_c_mean_anomaly'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[2,1].legend()
    axs[2,1].set_xlabel('mean_anomaly')
    if trace:
        axs[2,1].vlines(trace['c_mean_anomaly'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[2,2].hist(marginal_dists[i]['dist_c_eccentricity'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[2,2].legend()
    axs[2,2].set_xlabel('eccentricity')
    if trace:
        axs[2,2].vlines(trace['c_eccentricity'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[2,3].hist(marginal_dists[i]['dist_c_inclination'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[2,3].legend()
    axs[2,3].set_xlabel('inclination')
    if trace:
        axs[2,3].vlines(trace['c_inclination'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[3,0].hist(marginal_dists[i]['dist_c_argument_of_perigee'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[3,0].legend()
    axs[3,0].set_xlabel('argument_of_perigee')
    axs[3,0].set_ylabel('Chaser')
    if trace:
        axs[3,0].vlines(trace['c_argument_of_perigee'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[3,1].hist(marginal_dists[i]['dist_c_raan'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[3,1].legend()
    axs[3,1].set_xlabel('raan')
    if trace:
        axs[3,1].vlines(trace['c_raan'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[3,2].hist(marginal_dists[i]['dist_c_mean_motion_first_derivative'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[3,2].legend()
    axs[3,2].set_xlabel('mean_motion_first_derivative')
    if trace:
        axs[3,2].vlines(trace['c_mean_motion_first_derivative'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[3,3].hist(marginal_dists[i]['dist_c_b_star'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    # axs[3,3].legend()
    axs[3,3].set_xlabel('b_star')
    if trace:
        axs[3,3].vlines(trace['c_b_star'], 0, np.max(h)*1.05, linestyles='dashed')
#     ax.set_xlim(-0.01,0.01)


    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_mean_motion'].values_numpy()
        c = marginal_dists[i]['dist_c_mean_motion'].values_numpy()
        axs[4,0].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[4,0].set_xlabel('t_mean_motion')
    axs[4,0].set_ylabel('c_mean_motion')
    if trace:
        t = float(trace['t_mean_motion'])
        c = float(trace['c_mean_motion'])
        axs[4,0].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[4,0].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[4,0].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_mean_anomaly'].values_numpy()
        c = marginal_dists[i]['dist_c_mean_anomaly'].values_numpy()
        axs[4,1].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[4,1].set_xlabel('t_mean_anomaly')
    axs[4,1].set_ylabel('c_mean_anomaly')
    if trace:
        t = float(trace['t_mean_anomaly'])
        c = float(trace['c_mean_anomaly'])
        axs[4,1].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[4,1].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[4,1].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_eccentricity'].values_numpy()
        c = marginal_dists[i]['dist_c_eccentricity'].values_numpy()
        axs[4,2].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[4,2].set_xlabel('t_eccentricity')
    axs[4,2].set_ylabel('c_eccentricity')
    if trace:
        t = float(trace['t_eccentricity'])
        c = float(trace['c_eccentricity'])
        axs[4,2].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[4,2].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[4,2].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_inclination'].values_numpy()
        c = marginal_dists[i]['dist_c_inclination'].values_numpy()
        axs[4,3].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[4,3].set_xlabel('t_inclination')
    axs[4,3].set_ylabel('c_inclination')
    if trace:
        t = float(trace['t_inclination'])
        c = float(trace['c_inclination'])
        axs[4,3].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[4,3].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[4,3].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)


    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_argument_of_perigee'].values_numpy()
        c = marginal_dists[i]['dist_c_argument_of_perigee'].values_numpy()
        axs[5,0].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[5,0].set_xlabel('t_argument_of_perigee')
    axs[5,0].set_ylabel('c_argument_of_perigee')
    if trace:
        t = float(trace['t_argument_of_perigee'])
        c = float(trace['c_argument_of_perigee'])
        axs[5,0].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[5,0].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[5,0].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_raan'].values_numpy()
        c = marginal_dists[i]['dist_c_raan'].values_numpy()
        axs[5,1].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[5,1].set_xlabel('t_raan')
    axs[5,1].set_ylabel('c_raan')
    if trace:
        t = float(trace['t_raan'])
        c = float(trace['c_raan'])
        axs[5,1].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[5,1].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[5,1].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_mean_motion_first_derivative'].values_numpy()
        c = marginal_dists[i]['dist_c_mean_motion_first_derivative'].values_numpy()
        axs[5,2].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[5,2].set_xlabel('t_mean_motion_first_derivative')
    axs[5,2].set_ylabel('c_mean_motion_first_derivative')
    if trace:
        t = float(trace['t_mean_motion_first_derivative'])
        c = float(trace['c_mean_motion_first_derivative'])
        axs[5,2].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[5,2].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[5,2].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['dist_t_b_star'].values_numpy()
        c = marginal_dists[i]['dist_c_b_star'].values_numpy()
        axs[5,3].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[5,3].set_xlabel('t_b_star')
    axs[5,3].set_ylabel('c_b_star')
    if trace:
        t = float(trace['t_b_star'])
        c = float(trace['c_b_star'])
        axs[5,3].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[5,3].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[5,3].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    # Other variables from simulation
    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[6,0].hist(marginal_dists[i]['dist_time_min'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    axs[6,0].set_xlabel('time_min')
    if trace:
        axs[6,0].vlines(trace['time_min'], 0, np.max(h)*1.05, linestyles='dashed')

    ax = axs[6,1]
    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[6,1].hist(marginal_dists[i]['dist_d_min'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    axs[6,1].set_xlabel('d_min')
    if trace:
        axs[6,1].vlines(trace['d_min'], 0, np.max(h)*1.05, linestyles='dashed')

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        label = dists[i].name
        dist_conj = marginal_dists[i]['dist_conj']
        p_conj = sum(dist_conj.values)/len(dist_conj)
        axs[6,2].bar(['No conj', 'Conj'], [1-p_conj, p_conj], alpha=0.5)
    axs[6,2].set_xlabel('conj')
    if trace:
        axs[6,2].vlines(trace['conj']==1, 0, 1., linestyles='dashed')

    t_min, t_max = 1e30, -1e30
    c_min, c_max = 1e30, -1e30
    for i in range(len(dists)):
        t = marginal_dists[i]['d_conj'].values_numpy()
        c = marginal_dists[i]['d_min'].values_numpy()
        axs[6,3].scatter(x=t, y=c, alpha=0.5)
        t_min, t_max = min(t_min, t.min()), max(t_max, t.max())
        c_min, c_max = min(c_min, c.min()), max(c_max, c.max())
    axs[6,3].set_xlabel('d_conj')
    axs[6,3].set_ylabel('d_min')
    if trace:
        t = float(trace['d_conj'])
        c = float(trace['d_min'])
        axs[6,3].scatter(x=[t], y=[c], color='black')
        t_min, t_max = min(t_min, t), max(t_max, t)
        c_min, c_max = min(c_min, c), max(c_max, c)
    axs[6,3].set_xlim(t_min-(t_max-t_min)*0.05, t_max+(t_max-t_min)*0.05)
    axs[6,3].set_ylim(c_min-(c_max-c_min)*0.05, c_max+(c_max-c_min)*0.05)

    #axs[6,3].axis('off')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[7,0].hist(marginal_dists[i]['dist_time_conj'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    axs[7,0].set_xlabel('time_conj')
    if trace:
        if 'time_conj' in trace:
            if trace['time_conj'] is not None:
                axs[7,0].vlines(trace['time_conj'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[7,1].hist(marginal_dists[i]['dist_d_conj'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    axs[7,1].set_xlabel('d_conj')
    if trace:
        if 'd_conj' in trace:
            if trace['d_conj'] is not None:
                axs[7,1].vlines(trace['d_conj'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        h, _, _ = axs[7,2].hist(marginal_dists[i]['dist_num_cdms'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
    axs[7,2].set_xlabel('num_cdms')
    if trace:
        axs[7,2].vlines(trace['num_cdms'], 0, np.max(h)*1.05, linestyles='dashed')

    for i in range(len(dists)):
        label = dists[i].name
        dist_conj = marginal_dists[i]['dist_conj']
        if len(dist_conj) > 0:
            axs[7,3].hist(marginal_dists[i]['dist_time_cdm'].values_numpy(), bins=n_bins, alpha=0.5, density=True)
            axs[7,3].set_xlabel('time_cdm')

    plt.tight_layout()
    fig.legend()

    if file_name:
        print('Plotting to file: {}'.format(file_name))
        plt.savefig(file_name)
    return fig, axs

def plot_trace_orbit(trace, time_upsample_factor=100, figsize=(10, 8), file_name=None):
    t_color, c_color = 'red', 'forestgreen'

    time0 = float(trace['time0'])
    max_duration_days = float(trace['max_duration_days'])
    delta_time = float(trace['delta_time'])
    times = np.arange(time0, time0 + max_duration_days, delta_time)

    t_mean_motion = float(trace['t_mean_motion'])
    t_mean_motion_first_derivative = float(trace['t_mean_motion_first_derivative'])
    t_mean_motion_second_derivative= float(trace['t_mean_motion_second_derivative'])
    t_eccentricity = float(trace['t_eccentricity'])
    t_inclination = float(trace['t_inclination'])
    t_argument_of_perigee = float(trace['t_argument_of_perigee'])
    t_raan = float(trace['t_raan'])
    t_mean_anomaly = float(trace['t_mean_anomaly'])
    t_b_star = float(trace['t_b_star'])

    util.lpop_init(trace['t_tle0'])
    try:
        t_states = util.lpop_sequence_upsample(times, time_upsample_factor)
        t_prop_error = False
    except RuntimeError as e:
        t_prop_error = True

    c_mean_motion = float(trace['c_mean_motion'])
    c_mean_motion_first_derivative = float(trace['c_mean_motion_first_derivative'])
    c_mean_motion_second_derivative= float(trace['c_mean_motion_second_derivative'])
    c_eccentricity = float(trace['c_eccentricity'])
    c_inclination = float(trace['c_inclination'])
    c_argument_of_perigee = float(trace['c_argument_of_perigee'])
    c_raan = float(trace['c_raan'])
    c_mean_anomaly = float(trace['c_mean_anomaly'])
    c_b_star = float(trace['c_b_star'])

    util.lpop_init(trace['c_tle0'])
    try:
        c_states = util.lpop_sequence_upsample(times, time_upsample_factor)
        c_prop_error = False
    except RuntimeError as e:
        c_prop_error = True

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2,2,4, projection='3d')
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    earth_radius = 6.371e6
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    t_states[:,0] /= earth_radius
    c_states[:,0] /= earth_radius
    ax.plot_surface(x, y, z, color='blue', alpha=0.25)
    if not t_prop_error:
        ax.plot(t_states[:,0,0], t_states[:,0,1], t_states[:,0,2], alpha=0.75, color=t_color)
    if not c_prop_error:
        ax.plot(c_states[:,0,0], c_states[:,0,1], c_states[:,0,2], alpha=0.75, color=c_color)
#     set_axes_equal(ax)
    if trace['conj']:
        i_conj = int(trace['i_conj'])
        if not t_prop_error:
            t_pos_conj = t_states[i_conj, 0]
            ax.scatter(t_pos_conj[0], t_pos_conj[1], t_pos_conj[2], s=1e3, marker='*', color='green')
        if not c_prop_error:
            c_pos_conj = c_states[i_conj, 0]
            ax.scatter(c_pos_conj[0], c_pos_conj[1], c_pos_conj[2], s=1e3, marker='*', color='red')
#     ax.set_xlim(-12e6, 12e6)
#     ax.set_ylim(-12e6, 12e6)
#     ax.set_zlim(-12e6, 12e6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = fig.add_subplot(2,2,3)
    ax.plot(c_states[:,0,0], c_states[:,0,1],alpha=0.75, color=c_color)
    ax.plot(t_states[:,0,0], t_states[:,0,1],alpha=0.75, color=t_color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(color='lightgrey')
    ax = fig.add_subplot(2,2,1)
    ax.plot(c_states[:,0,0], c_states[:,0,2],alpha=0.75, color=c_color)
    ax.plot(t_states[:,0,0], t_states[:,0,2],alpha=0.75, color=t_color)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.grid(color='lightgrey')
    ax = fig.add_subplot(2,2,2)
    ax.plot(c_states[:,0,1], c_states[:,0,2],alpha=0.75, color=c_color)
    ax.plot(t_states[:,0,1], t_states[:,0,2],alpha=0.75, color=t_color)
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.grid(color='lightgrey')

    plt.tight_layout()

    if file_name:
        print('Plotting to file: {}'.format(file_name))
        plt.savefig(file_name)
    return ax

def plot_trace_event(trace, *args, **kwargs):
    event = util.trace_to_event(trace)
    return event.plot_features(*args, **kwargs)


def plot_combined(dists, trace, figsize=(20,10), file_name=None):
    file_name_1 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4())) + '.png'
    file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4())) + '.png'
    file_name_3 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4())) + '.png'

    plot_dist(dists, trace=trace, file_name=file_name_1)
    plot_trace_orbit(trace, file_name=file_name_2)
    features = ['MISS_DISTANCE', 'RELATIVE_SPEED', 'RELATIVE_POSITION_R', 'OBJECT1_CR_R', 'OBJECT1_CT_T', 'OBJECT1_CN_N', 'OBJECT1_CRDOT_RDOT', 'OBJECT1_CTDOT_TDOT', 'OBJECT1_CNDOT_NDOT', 'OBJECT2_CR_R', 'OBJECT2_CT_T', 'OBJECT2_CN_N', 'OBJECT2_CRDOT_RDOT', 'OBJECT2_CTDOT_TDOT', 'OBJECT2_CNDOT_NDOT']
    plot_trace_event(trace, features, file_name=file_name_3)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.09, wspace=0.05, left=0, right=1, bottom=0, top=1)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(mpimg.imread(file_name_1), interpolation='bicubic', aspect='auto')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mpimg.imread(file_name_2), interpolation='bicubic', aspect='auto')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(mpimg.imread(file_name_3), interpolation='bicubic', aspect='auto')
    ax.axis('off')
#     plt.tight_layout()

    if file_name is not None:
        print('Plotting combined plot to file: {}'.format(file_name))
        fig.savefig(file_name, dpi=150)
