#!/usr/bin/env python
# coding: utf-8

import EITfunc.EITlook as eit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cmx
import matplotlib.colorbar as cb
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

import os

from astropy import units as u
from astropy.constants import G
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.timeseries import LombScargle
import astropy.time
from astropy.coordinates import HeliocentricMeanEcliptic, HeliocentricTrueEcliptic, SkyCoord

from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body

import sunpy.coordinates

import scipy.optimize as sciop
from scipy.stats import mode, binned_statistic
import scipy.signal 

import time

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import pickle
import sys



wavelength = int(sys.argv[1])
corrstatus = str(sys.argv[2])

data = np.genfromtxt("./heliocentricLat_{0}_corrected_{1}.txt".format(wavelength, corrstatus))

badDataMask = (data[:,1] <=0.5) | (data[:,1] > 1.75)
data = data[~badDataMask]

if wavelength == 195:
    mask195oversampling = (data[:,0] > -3.25) & (data[:,0] <= -1.75)
    data = data[~mask195oversampling]
'''
print(np.shape(data))
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.errorbar(data[:,0],data[:,1],data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1)
ax.set_ylim(0,2)
plt.show()
'''
# uniform model
def uniform_ptransform(fitparam):
    """
    transforms a sample drawn from the unit cube [0, 1) to the appropriate uniform prior for the intercept of the flat line
    which I have chosen to be U(0.95, 1.05)
    """
    lineparam = fitparam*(1.05-0.95) + 0.95
    return lineparam

# linear model
def linear_ptransform(fitparams):
    """
    prior on slope is U(-0.025, 0.025)
    prior in intercept is U(0.95,1.05)
    """
    slope = fitparams[0]*(0.025 + 0.025) - 0.025
    intercept = fitparams[1]*(1.05-0.95) + 0.95
    return np.array((slope, intercept))

# linear in cos(lat) model
def linearcos_ptransform(fitparams):
    """
    prior on slope is U(-10,10)
    prior in intercept is U(0.95,1.05)
    """
    slope = fitparams[0]*(10 + 10) - 10
    intercept = fitparams[1]*(1.05-0.95) + 0.95
    return np.array((slope, intercept))

# gaussian log likelihood fn for each model
def uniform_log_likelihood(intercept, norm_flux=data[:,1], norm_fluxerr=data[:,2]):
    """
    following Hogg et al. 2010 chapter 7
    Assuming no covariance between data points and no uncertainty on latitude
    
    Inputs:
    intercept, y, yerr
                
    Returns:
    Log likelihood calculated for these parameters.
    """
    model_flux = intercept
    ll  = -np.sum(0.5*np.log(2*np.pi*norm_fluxerr**2)) - np.sum(((norm_flux - model_flux)**2)/(2*norm_fluxerr**2))
    return ll

def linear_log_likelihood(params, lat=np.abs(data[:,0]), norm_flux=data[:,1], norm_fluxerr=data[:,2]):
    """
    Inputs:
    params = (slope, intercept); flux, fluxerr
    """
    model_flux = params[0]*lat + params[1]
    ll  = -np.sum(0.5*np.log(2*np.pi*norm_fluxerr**2)) - np.sum(((norm_flux - model_flux)**2)/(2*norm_fluxerr**2))
    return ll

def linearcos_log_likelihood(params, coslat = (np.cos(data[:,0]*(np.pi/180.)) - np.cos(data[0,0]*(np.pi/180.))), norm_flux=data[:,1], norm_fluxerr=data[:,2]):
    """
    Inputs:
    params = (slope, intercept); flux, fluxerr
    """
    model_flux = params[0]*coslat + params[1]
    ll  = -np.sum(0.5*np.log(2*np.pi*norm_fluxerr**2)) - np.sum(((norm_flux - model_flux)**2)/(2*norm_fluxerr**2))
    return ll


def compareModels(modelNames, likelihoodFunctionList, priorTransformFunctionList, ndimList):
    results_dicts = []
    
    for i in range(len(modelNames)):
        name = modelNames[i]
        results_dict = {'model_name': name}
        sampler = dynesty.DynamicNestedSampler(loglikelihood=likelihoodFunctionList[i], prior_transform=priorTransformFunctionList[i], ndim=ndimList[i], nlive=1000)
        sampler.run_nested()
        sresults = sampler.results

        #tfig, taxes = dyplot.traceplot(sresults,show_titles=True)
        #cfig, caxes = dyplot.cornerplot(sresults,show_titles=True)
        results_dict['log_z'] = sresults['logz'][-1]
        results_dict['log_z_err'] = sresults['logzerr'][-1]

        samples = sresults.samples  # samples
        weights = sresults.importance_weights()

        # Compute 10%-90% quantiles.
        quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                     for samps in samples.T]
        
        # Compute weighted mean and covariance.
        mean, cov = dyfunc.mean_and_cov(samples, weights)
       
        results_dict['mean'] = mean
        results_dict['cov'] = cov
        
        # Resample weighted samples.
        samples_equal = sresults.samples_equal()

        results_dict['samples_equal'] = samples_equal

        picklefilename = "./EIT/dynestyrun_{0}_corrected_{1}_{2}.p".format(wavelength, corrstatus, name)
        pickle.dump(results_dict, open(picklefilename, 'wb'))

        results_dicts.append(results_dict)

    return results_dicts


def percentile16(x):
    return np.percentile(x,q=16)
def percentile84(x):
    return np.percentile(x,q=84)
    
def plot_uniform(data, mean, samples_equal):
    mean_model = mean[0]*np.ones_like(data[:,0])
    
    fig, axes = plt.subplots(2,1,figsize=(16,8))
    axes[0].errorbar(data[:,0],data[:,1],data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1,zorder=1)
    axes[0].axhline(1,ls=':',color='k',zorder=2)
    axes[0].plot(data[:,0],mean_model, 'r-', lw=2,zorder=3)
    for i in range(100):
        random_draw = np.random.choice(np.shape(samples_equal)[0])
        axes[0].plot(data[:,0], samples_equal[random_draw][0]*np.ones_like(data[:,0]),color='r',ls='-',lw=0.1,zorder=4)
    axes[0].set_ylim(0.75,1.25)
    
    axes[1].errorbar(data[:,0],data[:,1]-mean_model,data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1)
    axes[1].axhline(0,ls=':',color='k',zorder=2)
    axes[1].set_ylim(-0.25,0.25)

    binEdges = np.linspace(data[0,0], data[-1,0],100)
    fluxnorm_50 = binned_statistic(data[:,0], data[:,1], statistic='median',bins = binEdges)[0]
    fluxnorm_16 = binned_statistic(data[:,0], data[:,1], statistic=percentile16,bins = binEdges)[0]
    fluxnorm_84 = binned_statistic(data[:,0], data[:,1], statistic=percentile84,bins = binEdges)[0]

    le = fluxnorm_50 - fluxnorm_16
    ue = fluxnorm_84 - fluxnorm_50
    err2d  = np.vstack((le,ue))

    binCenters = (binEdges[0:-1] + binEdges[1:])/2.
    axes[0].errorbar(binCenters, fluxnorm_50, yerr=err2d, ls='None', elinewidth=1, capsize=0, marker='.',ms=15,markeredgecolor='k',c='#3772ff',ecolor='k',alpha=1,zorder=3) 
    mean_model_binned = mean[0]*np.ones_like(binCenters)
    axes[1].errorbar(binCenters, fluxnorm_50-mean_model_binned, yerr=err2d, ls='None', elinewidth=1, capsize=0, marker='.',ms=15,markeredgecolor='k',c='#3772ff',ecolor='k',alpha=1,zorder=3) 

    plt.show()
   
    return

def plot_linear(data, mean, samples_equal):
    mean_model = mean[0]*np.abs(data[:,0]) + mean[1]

    fig, axes = plt.subplots(2,1,figsize=(16,8))
    axes[0].errorbar(np.abs(data[:,0]),data[:,1],data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1,zorder=1)
    axes[0].axhline(1,ls=':',color='k',zorder=2)
    axes[0].plot(np.abs(data[:,0]),mean_model, 'r-', lw=2,zorder=3)
    for i in range(100):
        random_draw = np.random.choice(np.shape(samples_equal)[0])
        trial_model = samples_equal[random_draw][0]*np.abs(data[:,0]) + samples_equal[random_draw][1]
        axes[0].plot(np.abs(data[:,0]), trial_model,color='r',ls='-',lw=0.1,zorder=4)
    axes[0].set_ylim(0.75,1.25)
    
    axes[1].errorbar(np.abs(data[:,0]),data[:,1]-mean_model,data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1)
    axes[1].axhline(0,ls=':',color='k',zorder=2)
    axes[1].set_ylim(-0.25,0.25)

    binEdges = np.linspace(0, np.abs(data[-1,0]),100)
    fluxnorm_50 = binned_statistic(np.abs(data[:,0]), data[:,1], statistic='median',bins = binEdges)[0]
    fluxnorm_16 = binned_statistic(np.abs(data[:,0]), data[:,1], statistic=percentile16,bins = binEdges)[0]
    fluxnorm_84 = binned_statistic(np.abs(data[:,0]), data[:,1], statistic=percentile84,bins = binEdges)[0]

    le = fluxnorm_50 - fluxnorm_16
    ue = fluxnorm_84 - fluxnorm_50
    err2d  = np.vstack((le,ue))

    binCenters = (binEdges[0:-1] + binEdges[1:])/2.
    axes[0].errorbar(binCenters, fluxnorm_50, yerr=err2d, ls='None', elinewidth=1, capsize=0, marker='.',ms=15,markeredgecolor='k',c='#3772ff',ecolor='k',alpha=1,zorder=3) 
    mean_model_binned = mean[0]*np.abs(binCenters) + mean[1]
    axes[1].errorbar(binCenters, fluxnorm_50-mean_model_binned, yerr=err2d, ls='None', elinewidth=1, capsize=0, marker='.',ms=15,markeredgecolor='k',c='#3772ff',ecolor='k',alpha=1,zorder=3) 

    
    plt.show()

    return

def plot_linearcos(data, mean, samples_equal):
    x = np.cos(data[:,0]*(np.pi/180.))
    x0 = np.cos(data[0,0]*(np.pi/180.))
    
    mean_model = mean[0]*(x-x0) + mean[1]
    
    fig, axes = plt.subplots(2,1,figsize=(16,8))
    axes[0].errorbar(x,data[:,1],data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1,zorder=1)
    axes[0].axhline(1,ls=':',color='k',zorder=2)
    axes[0].plot(x,mean_model, 'r-', lw=2,zorder=3)
    
    for i in range(100):
        random_draw = np.random.choice(np.shape(samples_equal)[0])
        trial_model = samples_equal[random_draw][0]*(x-x0) + samples_equal[random_draw][1]
        axes[0].plot(x, trial_model,color='r',ls='-',lw=0.1,zorder=4)
    axes[0].set_ylim(0.75,1.25)
    
    axes[1].errorbar(x,data[:,1]-mean_model,data[:,2],marker='.',elinewidth=1,capsize=0,linestyle='None',color='#3772ff',alpha=0.1)
    axes[1].axhline(0,ls=':',color='k',zorder=2)
    axes[1].set_ylim(-0.25,0.25)

    binEdges = np.linspace(x0, 1,100)
    fluxnorm_50 = binned_statistic(x, data[:,1], statistic='median',bins = binEdges)[0]
    fluxnorm_16 = binned_statistic(x, data[:,1], statistic=percentile16,bins = binEdges)[0]
    fluxnorm_84 = binned_statistic(x, data[:,1], statistic=percentile84,bins = binEdges)[0]

    le = fluxnorm_50 - fluxnorm_16
    ue = fluxnorm_84 - fluxnorm_50
    err2d  = np.vstack((le,ue))

    binCenters = (binEdges[0:-1] + binEdges[1:])/2.
    axes[0].errorbar(binCenters, fluxnorm_50, yerr=err2d, ls='None', elinewidth=1, capsize=0, marker='.',ms=15,markeredgecolor='k',c='#3772ff',ecolor='k',alpha=1,zorder=3) 
    mean_model_binned = mean[0]*(binCenters-x0) + mean[1]
    axes[1].errorbar(binCenters, fluxnorm_50-mean_model_binned, yerr=err2d, ls='None', elinewidth=1, capsize=0, marker='.',ms=15,markeredgecolor='k',c='#3772ff',ecolor='k',alpha=1,zorder=3) 


    plt.show()
    return


def calc_uncertainty(sample_col):
    return 0.5*( (np.percentile(sample_col, 84) - np.percentile(sample_col, 50)) + (np.percentile(sample_col, 50) - np.percentile(sample_col, 16)))



modelNames = ['uniform','linear','linearcos']
results_dicts = compareModels(modelNames=modelNames,likelihoodFunctionList=[uniform_log_likelihood, linear_log_likelihood, linearcos_log_likelihood],priorTransformFunctionList=[uniform_ptransform, linear_ptransform, linearcos_ptransform],ndimList=[1,2,2])

LC = np.genfromtxt("./EIT/EIT{0}_LC_corrected_{1}.txt".format(wavelength, corrstatus))
metadata = Table.read("./EIT/EIT{0}_metadata.txt".format(wavelength), format='ascii')
cadence = (LC[:,0][1:] - LC[:,0][0:-1])


log_zs = []
for i in range(len(results_dicts)):
    log_zs.append(results_dicts[i]['log_z'])
log_zs = np.array(log_zs)
best_model = np.argmax(log_zs)

uniform_model = results_dicts[0]['mean'][0]*np.ones_like(metadata['HI-LAT-DEG'])
uniform_model_b_unc = calc_uncertainty(results_dicts[0]['samples_equal'])


linear_model = results_dicts[1]['mean'][0]*np.abs(metadata['HI-LAT-DEG']) + results_dicts[1]['mean'][1]
linear_model_m_unc = calc_uncertainty(results_dicts[1]['samples_equal'][:,0])
linear_model_b_unc = calc_uncertainty(results_dicts[1]['samples_equal'][:,1])

# we use the 0th entry in the latitude column because the data have already been sorted in order of increasing latitude
x0 = np.cos(data[0,0]*(np.pi/180.))
linearcos_model = results_dicts[2]['mean'][0] * (np.cos(metadata['HI-LAT-DEG']*(np.pi/180.)) - x0) + results_dicts[2]['mean'][1]
linearcos_model_m_unc = calc_uncertainty(results_dicts[2]['samples_equal'][:,0])
linearcos_model_b_unc = calc_uncertainty(results_dicts[2]['samples_equal'][:,1])

print("uniform model")
print("{0} +/- {1}".format(results_dicts[0]['mean'][0], uniform_model_b_unc))
print("")

print("linear model unc")
print("{0} +/- {1}".format(results_dicts[1]['mean'][0], linear_model_m_unc))
print("{0} +/- {1}".format(results_dicts[1]['mean'][1], linear_model_b_unc))
print("")

print("linearcos model unc")
print("{0} +/- {1}".format(results_dicts[2]['mean'][0], linearcos_model_m_unc))
print("{0} +/- {1}".format(results_dicts[2]['mean'][1], linearcos_model_b_unc))
print("")

if best_model==0:
    print('uniform best')
    LC_corr = LC[:,0]/linear_model
    LC_unc_corr = np.sqrt( (LC[:,2]/uniform_model)**2 + (uniform_model_b_unc**2 * ((LC[:,1])/uniform_model**2)**2) )
    plot_uniform(data,mean=results_dicts[0]['mean'],samples_equal=results_dicts[0]['samples_equal'])
elif best_model==1:
    print('linear best')
    LC_corr = LC[:,1]/linear_model
    LC_unc_corr = np.sqrt( (LC[:,2]/linear_model)**2 + (linear_model_m_unc**2 * ((LC[:,1]*np.abs(metadata['HI-LAT-DEG']))/linear_model**2)**2 ) + (linear_model_b_unc**2 * ((LC[:,1])/linear_model**2)**2) )
    plot_linear(data,mean=results_dicts[1]['mean'],samples_equal=results_dicts[1]['samples_equal'])
elif best_model==2:
    print('linearcos best')
    LC_corr = LC[:,1]/linearcos_model
    LC_unc_corr = np.sqrt( (LC[:,2]/linearcos_model)**2 + (linearcos_model_m_unc**2 * ((LC[:,1]*np.cos(metadata['HI-LAT-DEG']*(np.pi/180.)))/linearcos_model**2)**2 ) + (linearcos_model_b_unc**2 * ((LC[:,1])/linearcos_model**2)**2) )
    plot_linearcos(data,mean=results_dicts[2]['mean'],samples_equal=results_dicts[2]['samples_equal'])

toSave = np.vstack((LC[:,0], LC_corr, LC_unc_corr, LC[:,3], LC[:,4])).T
header = 'average_observation_time[JD] flux[DN/s] flux_unc[DN/s] last_data_point_before_bakeout first_data_point_after_bakeout'
np.savetxt("./EIT/EIT{0}_LC_corrected_{1}_heliocentriccorr_{2}.txt".format(wavelength, corrstatus, modelNames[best_model]), toSave, fmt='%f %f %f %d %d', delimiter=' ', header=header)


