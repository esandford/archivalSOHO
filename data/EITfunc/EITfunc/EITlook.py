import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cmx
import matplotlib.colorbar as cb
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

import os
import copy

from astropy import units as u
from astropy.constants import G
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.timeseries import LombScargle
import astropy.time

import scipy.optimize as sciop
from scipy.stats import mode, binned_statistic

import time

__all__ = ['normal_equation', 'draw_solar_rot', 'plot_full_sun_EIT', 'plot_full_sun_annuli_EIT', 'plot_flux_extrapolation_EIT', 'plot_other_EIT', 'make_disk_model', 'make_disk_corona_model','chi_squared','fit_solar_center','find_solar_center','findCoL','center_from_header']

def normal_equation(X,order,Y,Yerr):
    """
    Solve the normal equation B = (X.T*C.inv*X).inv*X.T*C.inv*Y to fit a linear model to data
    Inputs:
    X = matrix of x values
    order = integer polynomial order (1 for flat line, 2 for linear, 3 for quadratic, etc)
    Y = vector of y values
    Yerr = vector of yerr values
    Outputs:
    B = vector of model parameters that minimizes chi^2
    """
    
    X = np.vander(X, order)
    
    XTX = np.dot(X.T, X/Yerr[:, None]**2)
    
    B = np.linalg.solve(XTX, np.dot(X.T, Y/Yerr**2))
    
    return B

def interpolate_darkcurrent(obsTime, darkFramesArr):
    dark_times = darkFramesArr[:,0]
    dark_fluxes = darkFramesArr[:,1]
    dark_uncs = darkFramesArr[:,2]

    i_prev = np.arange(len(darkFramesArr))[dark_times < obsTime][-1]
    i_next = i_prev + 1

    interp_dark_flux = dark_fluxes[i_prev] + ((dark_fluxes[i_next] - dark_fluxes[i_prev])*((obsTime - dark_times[i_prev])/(dark_times[i_next]-dark_times[i_prev])))

    return interp_dark_flux


def draw_solar_rot(ax, center, radius, facecolor='k', edgecolor='None', theta1=135, theta2=30):
    
    # Add the ring
    rwidth = radius/6
    ring = patches.Wedge(center, radius, theta1, theta2, width=rwidth)
    # Triangle edges
    offset = radius/3
    xcent  = center[0] + radius - (rwidth/2)
    left   = [xcent - offset - radius/10, center[1]+radius/5]
    right  = [xcent + offset, center[1]+radius/3]

    top = [xcent-7, center[1]+radius]
    arrow  = plt.Polygon([left, right, top, left])
    p = PatchCollection(
        [ring, arrow], 
        edgecolor = edgecolor, 
        facecolor = facecolor
    )
    ax.add_collection(p)


def plot_full_sun_EIT(fitsFileName, header, data, xcenter=512, ycenter=512, save=False, saveFileName=None):

    fig = plt.figure(figsize=(24,8))
    gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
        
    cs = ax0.imshow(np.log10(data),cmap='Greys_r',interpolation='None',origin="lower")
    ax0.set_title('{0}, {1} angstroms,\n{2}'.format(header['SCI_OBJ'],header['WAVELNTH'],fitsFileName))
    
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad="2%")
    fig.add_axes(cax)
    fig.colorbar(cs, cax=cax,label="log_10(count)")

    rwidth = np.max(np.array((np.abs(xcenter-512),np.abs(ycenter-512)))) + 512

    rect0 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor='y', facecolor='none',angle=180.,rotation_point='xy')
    rect1 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor='m', facecolor='none')
    rect2 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor='r', facecolor='none',angle=270.,rotation_point='xy')
    rect3 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor='b', facecolor='none',angle=90.,rotation_point='xy')
        
    rect4 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor='k', facecolor='none',angle=270+45.,rotation_point='xy')
    rect5 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor='darkorange', facecolor='none',angle=180+45.,rotation_point='xy')
    rect6 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor='g', facecolor='none',angle=90+45.,rotation_point='xy')
    rect7 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor='c', facecolor='none',angle=45.,rotation_point='xy')
        
    for r in [rect0,rect1,rect2,rect3,rect4,rect5,rect6,rect7]:
        ax0.add_patch(r)
    
    draw_solar_rot(ax=ax0, center=(xcenter,950), radius=30, facecolor='k', edgecolor='k', theta1=135, theta2=30) 

    ax0.set_xlim(0,1024)
    ax0.set_ylim(0,1024)
    
    try:
        ax1.plot(np.arange(xcenter),data[ycenter,0:xcenter][::-1],'y-')
        ax1.plot(np.arange(1024-xcenter),data[ycenter,xcenter:],'m-')
        ax1.plot(np.arange(ycenter),data[0:ycenter,xcenter][::-1],'r-')
        ax1.plot(np.arange(1024-ycenter),data[ycenter:,xcenter],'b-')
        ax1.axvline(380,color='k',ls=':')
        ax1.axvline(512,color='k',ls='-')
        ax1.axhline(1,color='k',ls=':')


        ur_length = np.min(np.array((1024-ycenter, 1024-xcenter)))
        lr_length = np.min(np.array((1024-xcenter, ycenter)))
        ll_length = np.min(np.array((ycenter, xcenter)))
        ul_length = np.min(np.array((xcenter, 1024-ycenter)))
        ur = np.zeros(ur_length)
        lr = np.zeros(lr_length)
        ll = np.zeros(ll_length)
        ul = np.zeros(ul_length)

        for i in range(ur_length):
            ur[i] = data[ycenter+i,xcenter+i]

        for i in range(lr_length):
            lr[i] = data[ycenter-i,xcenter+i]

        for i in range(ll_length):
            ll[i] = data[ycenter-i,xcenter-i]

        for i in range(ul_length):
            ul[i] = data[ycenter+i,xcenter-i]
        
        ax1.plot(np.arange(lr_length)*np.sqrt(2),lr,'k-')
        ax1.plot(np.arange(ll_length)*np.sqrt(2),ll,color='darkorange',ls='-')
        ax1.plot(np.arange(ul_length)*np.sqrt(2),ul,'g-')
        ax1.plot(np.arange(ur_length)*np.sqrt(2),ur,'c-')
            
        ax1.set_ylim(1.e-2,2.e3)
        ax1.set_yscale("log")
        ax1.set_xlim(-1,512*np.sqrt(2))
        ax1.set_xlabel("distance from center [pixels]")
        ax1.set_ylabel("count")

    except IndexError:
        pass
        
    plt.subplots_adjust(wspace=0.275)

    if save is True:
        plt.savefig("{0}".format(saveFileName),bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return

def detect_pinholes(data, mask=False):
    """
    return True if the bright spots on the CCD are present in this image.
    """
    image_idxs = np.indices((1024, 1024))
    
    brightspot1_mask = ( ((image_idxs[0,:,:] - 1022)**2 + (image_idxs[1,:,:] - 585)**2) <= 200.**2 ) & ( image_idxs[0,:,:] > 931 ) & ~np.isnan(data)
    brightspot2_mask = ( ((image_idxs[0,:,:] - 1022)**2 + (image_idxs[1,:,:] - 130)**2) <= 150.**2 ) & ( image_idxs[0,:,:] > 931 ) & ~np.isnan(data)
    brightspot1_comp_mask = ( ((image_idxs[0,:,:] - 1)**2 + (image_idxs[1,:,:] - 585)**2) <= 200.**2 ) & ( image_idxs[0,:,:] < 92 ) & ~np.isnan(data)
    brightspot2_comp_mask = ( ((image_idxs[0,:,:] - 1)**2 + (image_idxs[1,:,:] - 130)**2) <= 150.**2 ) & ( image_idxs[0,:,:] < 92 ) & ~np.isnan(data)
    brightspot1small_mask = ( ((image_idxs[0,:,:] - 1016)**2 + (image_idxs[1,:,:] - 557)**2) <= 40.**2 ) & ~np.isnan(data)
    brightspot1small_comp_mask = ( ((image_idxs[0,:,:] - 7)**2 + (image_idxs[1,:,:] - 557)**2) <= 40.**2 ) & ~np.isnan(data)

    bs1 = False
    bs1small = False
    bs2 = False

    """
    print("bright spot 1")
    print("spot mean is {0}".format(np.mean(data[brightspot1_mask])))
    print("comp mean is {0}".format(np.mean(data[brightspot1_comp_mask])))
    
    print("bright spot 1 small")
    print("spot mean is {0}".format(np.mean(data[brightspot1small_mask])))
    print("comp mean is {0}".format(np.mean(data[brightspot1small_comp_mask])))
    
    print("bright spot 2")
    print("spot mean is {0}".format(np.mean(data[brightspot2_mask])))
    print("comp mean is {0}".format(np.mean(data[brightspot2_comp_mask])))
    """

    if np.mean(data[brightspot1_mask]) > 2.*np.mean(data[brightspot1_comp_mask]):
        bs1 = True

    if np.mean(data[brightspot1small_mask]) > 2.*np.mean(data[brightspot1small_comp_mask]):
        bs1small = True

    if np.mean(data[brightspot2_mask]) > 2.*np.mean(data[brightspot2_comp_mask]):
        bs2 = True

    return bs1, bs1small, bs2

def mask_pinholes(data, bs1, bs1small, bs2):
    """
    return True if the bright spots on the CCD are present in this image.
    """
    image_idxs = np.indices((1024, 1024))
    
    brightspot1_mask = ( ((image_idxs[0,:,:] - 1022)**2 + (image_idxs[1,:,:] - 585)**2) <= 200.**2 ) & ( image_idxs[0,:,:] > 931 ) & ~np.isnan(data)
    brightspot2_mask = ( ((image_idxs[0,:,:] - 1022)**2 + (image_idxs[1,:,:] - 130)**2) <= 150.**2 ) & ( image_idxs[0,:,:] > 931 ) & ~np.isnan(data)
    brightspot1_comp_mask = ( ((image_idxs[0,:,:] - 1)**2 + (image_idxs[1,:,:] - 585)**2) <= 200.**2 ) & ( image_idxs[0,:,:] < 92 ) & ~np.isnan(data)
    brightspot2_comp_mask = ( ((image_idxs[0,:,:] - 1)**2 + (image_idxs[1,:,:] - 130)**2) <= 150.**2 ) & ( image_idxs[0,:,:] < 92 ) & ~np.isnan(data)
    brightspot1small_mask = ( ((image_idxs[0,:,:] - 1016)**2 + (image_idxs[1,:,:] - 557)**2) <= 40.**2 ) & ~np.isnan(data)
    brightspot1small_comp_mask = ( ((image_idxs[0,:,:] - 7)**2 + (image_idxs[1,:,:] - 557)**2) <= 40.**2 ) & ~np.isnan(data)


    maskedData = copy.deepcopy(data)
    
    if bs1 is True:
        maskedData[brightspot1_mask] = np.nan
    if bs1small is True:
        maskedData[brightspot1small_mask] = np.nan
    if bs2 is True:
        maskedData[brightspot2_mask] = np.nan

    #if both of these are true just exclude the entire top of image
    if bs1 is True and bs2 is True:
        maskedData[( image_idxs[0,:,:] > 931 )] = np.nan
    
    return maskedData
    
def flux_in_annuli(data, darkFlux, xcenter, ycenter, rExtrapolate=512):
    #calculate flux in annuli of increasing radius

    image_idxs = np.indices((1024, 1024))
            
    rs = np.arange(0., np.round(512.*np.sqrt(2),0))
    annNpixs = np.zeros_like(rs)
    annNpixs_unc = np.zeros_like(rs)

    annFluxes = np.zeros_like(rs)
    annFluxes_unc = np.zeros_like(rs)

    for i in range(rExtrapolate):
        r = rs[i]

        annulus_mask = ( ((image_idxs[0,:,:] - ycenter)**2 + (image_idxs[1,:,:] - xcenter)**2) > (r-0.5)**2 ) & ( ((image_idxs[0,:,:] - ycenter)**2 + (image_idxs[1,:,:] - xcenter)**2) <= (r+0.5)**2 ) & ~np.isnan(data)

        annNpix = np.sum(np.ones_like(data)[annulus_mask])
        annNpixs[i] = annNpix
        annNpixs_unc[i] = np.sqrt(annNpix)

        annFlux = np.sum(data[annulus_mask])
        annFluxes[i] = annFlux
        # the below is the previous wrong version, when i thought that the dark frames were measuring thermal noise rather than analog-to-digital converter offset
        #annFluxes_unc[i] = np.sqrt(annFlux + (annNpix * darkFlux))
        annFluxes_unc[i] = np.sqrt(annFlux)

    for i in range(rExtrapolate, int(np.round(512.*np.sqrt(2),0))):
        r = rs[i]

        annulus_mask = ( ((image_idxs[0,:,:] - ycenter)**2 + (image_idxs[1,:,:] - xcenter)**2) > (r-0.5)**2 ) & ( ((image_idxs[0,:,:] - ycenter)**2 + (image_idxs[1,:,:] - xcenter)**2) <= (r+0.5)**2 ) & ~np.isnan(data)

        pred_Npix = 2*np.pi*r
        actual_Npix = np.sum(np.ones_like(data)[annulus_mask])

        annNpixs[i] = pred_Npix
        annNpixs_unc[i] = np.sqrt(pred_Npix)

        annFlux_known = np.sum(data[annulus_mask])
        annFlux_known_unc = np.sqrt(annFlux_known + (actual_Npix * darkFlux))
        
        annFlux_tot = annFlux_known * (pred_Npix/actual_Npix) #decided to keep this mean rather than median because it makes the flux_in_annulus vs. r curves smoother over rExtrapolate
        #using standard error propagation formula
        annFlux_tot_unc = np.sqrt( ((pred_Npix/actual_Npix)**2 * annFlux_known_unc**2) + ((annFlux_known/actual_Npix)**2 * pred_Npix ) )

        annFluxes[i] = annFlux_tot
        annFluxes_unc[i] = annFlux_tot_unc

    return rs, annNpixs, annNpixs_unc, annFluxes, annFluxes_unc

def overall_flux(annFluxes, annFluxes_unc):
    overallFlux = np.sum(annFluxes)
    overallFlux_unc = np.sqrt(np.sum(annFluxes_unc**2))
    return overallFlux, overallFlux_unc

def image_to_LCpoint(data, darkFlux, xcenter=512, ycenter=512, maskBrightSpots=False):
    
    if maskBrightSpots is True:
        bs1, bs1small, bs2 = detect_pinholes(data,mask=True)
        if np.any((bs1, bs1small, bs2)):
            data = mask_pinholes(data, bs1, bs1small, bs2)
            
    rs, annNpixs, annNpixs_unc, annFluxes, annFluxes_unc = flux_in_annuli(data, darkFlux, xcenter, ycenter)

    overallFlux, overallFlux_unc = overall_flux(annFluxes, annFluxes_unc)

    return overallFlux, overallFlux_unc

def plot_full_sun_annuli_EIT(fitsFileName, header, data, darkFlux, figtitle, xcenter=512, ycenter=512, save=False, saveFileName=None, maskBrightSpots=False, plotBrightSpotMasks=False):

    rs, annNpixs, annNpixs_unc, annFluxes, annFluxes_unc = flux_in_annuli(data, darkFlux, xcenter, ycenter, rExtrapolate=512)
    plotMasked = False

    if maskBrightSpots is True:
        bs1, bs1small, bs2 = detect_pinholes(data,mask=True)
        if np.any((bs1, bs1small, bs2)):
            plotMasked = True
            maskedData = mask_pinholes(data, bs1, bs1small, bs2)
            rs, masked_annNpixs, masked_annNpixs_unc, masked_annFluxes, masked_annFluxes_unc = flux_in_annuli(maskedData, darkFlux, xcenter, ycenter, rExtrapolate=512)

    
    fig = plt.figure(figsize=(24,8))
    gs = gridspec.GridSpec(ncols=12, nrows=1, figure=fig)
    ax0 = fig.add_subplot(gs[0:5])
    ax1 = fig.add_subplot(gs[0:6, 6:])
    
    cs = ax0.imshow(np.log10(data),cmap='Greys_r',interpolation='None',origin="lower")
    ax0.set_title(figtitle,fontsize=24,pad=10)
    ax0.set_xticks([0,512,1024])
    ax0.set_yticks([0,512,1024])
    ax0.tick_params(axis='both', which='major', labelsize=20)
    ax0.set_xlabel('pixel index', fontsize=24)
    ax0.set_ylabel('pixel index', fontsize=24)
    
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad="2%")
    fig.add_axes(cax)
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label(label="flux [DN/s]",fontsize=24)
    cax.set_yticks([-1, 0, 1, 2])
    cax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    cax.tick_params(axis='both', which='major', labelsize=26) 

    sliceColors = ['r', 'darkorange', 'y', 'g', 'b', 'c', 'm', '#432371']

    rwidth = np.max(np.array((np.abs(xcenter-512),np.abs(ycenter-512)))) + 512

    rect0 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[0], facecolor='none',angle=0.,rotation_point='xy')
    rect2 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[2], facecolor='none',angle=90.,rotation_point='xy')
    rect4 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[4], facecolor='none',angle=180.,rotation_point='xy')
    rect6 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[6], facecolor='none',angle=270.,rotation_point='xy')
    
    rect1 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[1], facecolor='none',angle=0+45.,rotation_point='xy')
    rect3 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[3], facecolor='none',angle=90+45.,rotation_point='xy')
    rect5 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[5], facecolor='none',angle=180+45.,rotation_point='xy')
    rect7 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[7], facecolor='none',angle=270+45.,rotation_point='xy')

    for r in [rect0,rect1,rect2,rect3,rect4,rect5,rect6,rect7]:
        ax0.add_patch(r)
    
    draw_solar_rot(ax=ax0, center=(xcenter,950), radius=30, facecolor='k', edgecolor='k', theta1=135, theta2=30) 

    ax0.set_xlim(0,1024)
    ax0.set_ylim(0,1024)

    if plotBrightSpotMasks is True:
        circ1 = plt.Circle((557,1016),40,color='#FFFFFF',fill=False)
        circ2 = plt.Circle((130,1022),150,color='#FFFFFF',fill=False)
        circ3 = plt.Circle((585,1022),200,color='#FFFFFF',fill=False)
        ax0.axhline(931,color='#FFFFFF')
        ax0.add_patch(circ1)
        ax0.add_patch(circ2)
        ax0.add_patch(circ3)

    
    ax1.axvline(512,color='k',ls=':')
    ax1.plot(np.arange(xcenter),data[ycenter,0:xcenter][::-1],ls='-',color=sliceColors[4])
    ax1.plot(np.arange(1024-xcenter),data[ycenter,xcenter:],  ls='-',color=sliceColors[0])
    ax1.plot(np.arange(ycenter),data[0:ycenter,xcenter][::-1],ls='-',color=sliceColors[6])
    ax1.plot(np.arange(1024-ycenter),data[ycenter:,xcenter],  ls='-',color=sliceColors[2])

    ur_length = np.min(np.array((1024-ycenter, 1024-xcenter)))
    lr_length = np.min(np.array((1024-xcenter, ycenter)))
    ll_length = np.min(np.array((ycenter, xcenter)))
    ul_length = np.min(np.array((xcenter, 1024-ycenter)))
    ur = np.zeros(ur_length)
    lr = np.zeros(lr_length)
    ll = np.zeros(ll_length)
    ul = np.zeros(ul_length)

    for i in range(ur_length):
        ur[i] = data[ycenter+i,xcenter+i]

    for i in range(lr_length):
        lr[i] = data[ycenter-i,xcenter+i]

    for i in range(ll_length):
        ll[i] = data[ycenter-i,xcenter-i]

    for i in range(ul_length):
        ul[i] = data[ycenter+i,xcenter-i]
        
    ax1.plot(np.arange(lr_length)*np.sqrt(2),lr,ls='-',color=sliceColors[7])
    ax1.plot(np.arange(ll_length)*np.sqrt(2),ll,ls='-',color=sliceColors[5])
    ax1.plot(np.arange(ul_length)*np.sqrt(2),ul,ls='-',color=sliceColors[3])
    ax1.plot(np.arange(ur_length)*np.sqrt(2),ur,ls='-',color=sliceColors[1])
            
    ax1.set_yscale("log")
    ax1.set_xticks([0,100,200,300,400,512,724])
    ax1.set_xlabel("distance from center [pixel side lengths]",fontsize=24)
    ax1.set_ylabel("flux [DN/s]",fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    if plotMasked is True:
        ax1.errorbar(rs, annFluxes,yerr=annFluxes_unc, capsize=0, color='k', elinewidth=0.5,alpha=0.5,label='total flux in annulus')
        ax1.errorbar(rs, masked_annFluxes,yerr=masked_annFluxes_unc, capsize=0, color='b', elinewidth=0.5,alpha=0.5)
    else:
        #ax1.errorbar(rs, annFluxes,yerr=annFluxes_unc, capsize=0, color='k', elinewidth=0.5,alpha=1,label='total flux in annulus')
        ax1.plot(rs, annFluxes,color='k', alpha=1,label='total flux in annulus',zorder=2)
        ax1.fill_between(rs, annFluxes-annFluxes_unc, annFluxes+annFluxes_unc,color='k',edgecolor='None',alpha=0.4,zorder=1)
    
    ax1.set_xlim(-1,512*np.sqrt(2))
    ax1.legend(loc='upper right',frameon=False,prop={'size': 20})


    
    plt.subplots_adjust(wspace=1)

    if save is True:
        plt.savefig("{0}".format(saveFileName),bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return

def plot_flux_extrapolation_EIT(fitsFileName, header, data, darkFlux, figtitle, xcenter=512, ycenter=512, save=False, saveFileName=None):

    #calculate flux in annuli of increasing radius
    image_idxs = np.indices((1024, 1024))
            
    rs = np.arange(0., 512.)
    annFluxes = np.zeros_like(rs)
    annNpixs = np.zeros_like(rs)
    annFluxes_unc = np.zeros_like(rs)
    for i, r in enumerate(rs):
                
        annulus_mask = ( ((image_idxs[0,:,:] - ycenter)**2 + (image_idxs[1,:,:] - xcenter)**2) > (r-0.5)**2 ) & ( ((image_idxs[0,:,:] - ycenter)**2 + (image_idxs[1,:,:] - xcenter)**2) <= (r+0.5)**2 ) & ~np.isnan(data)

        annFluxes[i] = np.sum(data[annulus_mask])
        annNpix = np.sum(np.ones_like(data)[annulus_mask])
        annNpixs[i] = annNpix
        annFluxes_unc[i] = np.sqrt(np.sum(data[annulus_mask]) + (annNpix * darkFlux))

    fig = plt.figure(figsize=(24,24))
    gs = gridspec.GridSpec(ncols=12, nrows=16, figure=fig)
    ax0 = fig.add_subplot(gs[:, 0:5])
    ax1 = fig.add_subplot(gs[0:6, 6:])
    ax1residuals = fig.add_subplot(gs[6:8, 6:])
    ax2 = fig.add_subplot(gs[8:14, 6:]) 
    ax2residuals = fig.add_subplot(gs[14:16,6:])

    cs = ax0.imshow(np.log10(data),cmap='Greys_r',interpolation='None',origin="lower")
    ax0.set_title(figtitle,fontsize=20,pad=10)
    ax0.set_xticks([0,512,1024])
    ax0.set_yticks([0,512,1024])
    ax0.tick_params(axis='both', which='major', labelsize=16)
    
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad="2%")
    fig.add_axes(cax)
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label(label=r"$\log_{10}$(count rate [DN/s])",fontsize=18)
    cax.tick_params(labelsize=16) 

    sliceColors = ['r', 'darkorange', 'y', 'g', 'b', 'c', 'm', 'k']

    rwidth = np.max(np.array((np.abs(xcenter-512),np.abs(ycenter-512)))) + 512

    rect0 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[0], facecolor='none',angle=0.,rotation_point='xy')
    rect2 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[2], facecolor='none',angle=90.,rotation_point='xy')
    rect4 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[4], facecolor='none',angle=180.,rotation_point='xy')
    rect6 = patches.Rectangle((xcenter, ycenter), width=rwidth, height = 1, linewidth=1, edgecolor=sliceColors[6], facecolor='none',angle=270.,rotation_point='xy')
    
    rect1 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[1], facecolor='none',angle=0+45.,rotation_point='xy')
    rect3 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[3], facecolor='none',angle=90+45.,rotation_point='xy')
    rect5 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[5], facecolor='none',angle=180+45.,rotation_point='xy')
    rect7 = patches.Rectangle((xcenter, ycenter), width=rwidth*np.sqrt(2), height = 1, linewidth=1, edgecolor=sliceColors[7], facecolor='none',angle=270+45.,rotation_point='xy')

    for r in [rect0,rect1,rect2,rect3,rect4,rect5,rect6,rect7]:
        ax0.add_patch(r)
    
    draw_solar_rot(ax=ax0, center=(xcenter,950), radius=30, facecolor='k', edgecolor='k', theta1=135, theta2=30) 

    ax0.set_xlim(0,1024)
    ax0.set_ylim(0,1024)
    
    ax1.plot(np.arange(xcenter),data[ycenter,0:xcenter][::-1],ls='-',color=sliceColors[4])
    ax1.plot(np.arange(1024-xcenter),data[ycenter,xcenter:],  ls='-',color=sliceColors[0])
    ax1.plot(np.arange(ycenter),data[0:ycenter,xcenter][::-1],ls='-',color=sliceColors[6])
    ax1.plot(np.arange(1024-ycenter),data[ycenter:,xcenter],  ls='-',color=sliceColors[2])

    ur_length = np.min(np.array((1024-ycenter, 1024-xcenter)))
    lr_length = np.min(np.array((1024-xcenter, ycenter)))
    ll_length = np.min(np.array((ycenter, xcenter)))
    ul_length = np.min(np.array((xcenter, 1024-ycenter)))
    ur = np.zeros(ur_length)
    lr = np.zeros(lr_length)
    ll = np.zeros(ll_length)
    ul = np.zeros(ul_length)

    for i in range(ur_length):
        ur[i] = data[ycenter+i,xcenter+i]

    for i in range(lr_length):
        lr[i] = data[ycenter-i,xcenter+i]

    for i in range(ll_length):
        ll[i] = data[ycenter-i,xcenter-i]

    for i in range(ul_length):
        ul[i] = data[ycenter+i,xcenter-i]
        
    ax1.plot(np.arange(lr_length)*np.sqrt(2),lr,ls='-',color=sliceColors[7])
    ax1.plot(np.arange(ll_length)*np.sqrt(2),ll,ls='-',color=sliceColors[5])
    ax1.plot(np.arange(ul_length)*np.sqrt(2),ul,ls='-',color=sliceColors[3])
    ax1.plot(np.arange(ur_length)*np.sqrt(2),ur,ls='-',color=sliceColors[1])
            
    ax1.set_yscale("log")
    ax1.set_xticklabels([])
    #ax1.set_xlabel("distance from center [pixels]",fontsize=18)
    ax1.set_ylabel("count rate [DN/s]"+"\n"+"along slice",fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    for i, dataArray in enumerate([ur, ul, ll, lr]):
        #eliminate nans--this makes the below indexing very confusing
        xs = np.arange(len(dataArray),dtype=float)

        noNans = ~np.isnan(dataArray)
        dataArray = dataArray[noNans]
        xs = xs[noNans]

        coronaMask = (xs > 350./np.sqrt(2))
        peakIdx = int(np.round(350/np.sqrt(2),0)) + np.argmax(dataArray[coronaMask])
        
        #ax1.axvline(peakIdx*np.sqrt(2))
        #peakIdx = xs[np.argmax(dataArray)] # this will be smaller than ~380 because we haven't applied the *sqrt(2) transformation
  
        startIdx = int(peakIdx + 0.7*((511/np.sqrt(2)) - peakIdx))
        endIdx = int(peakIdx + 0.9*((511/np.sqrt(2)) - peakIdx))

        window_to_fit = (xs >= startIdx) & (xs < endIdx)
        window_to_plot = (xs >= startIdx)

        xs_fit = xs[window_to_fit]
        xs_fit_scaled = (xs_fit - (startIdx-1))/(endIdx-startIdx) 

        xs_to_plot = xs[window_to_plot]
        xs_to_plot_scaled = (xs_to_plot - (startIdx-1))/(endIdx-startIdx) 

        try:

            best_params_exp = normal_equation(X=xs_fit_scaled,
                                              order=2,
                                              Y=np.log(dataArray[window_to_fit]),
                                              Yerr=np.ones_like(xs_fit_scaled))
            
            model_to_plot = np.exp(best_params_exp[1])*np.exp(xs_to_plot_scaled)**best_params_exp[0]
            #model_to_plot = np.exp(best_params_exp[2]) * np.exp(xs_to_plot_scaled)**best_params_exp[1] * np.exp(xs_to_plot_scaled**2)**best_params_exp[0]

            '''
            best_params_pl = normal_equation(X=np.log(xs_fit_scaled),
                                            order=3,
                                            Y=np.log(dataArray[window_to_fit]),
                                            Yerr=np.ones_like(xs_fit_scaled))
            
            #model_to_plot = np.exp(best_params_pl[1]) * xs_to_plot_scaled**(best_params_pl[0])
            model_to_plot = np.exp(best_params_pl[2]) * xs_to_plot_scaled**(2*best_params_pl[0]+best_params_pl[1])
            
            '''
            ax1.plot(xs_to_plot*np.sqrt(2), model_to_plot,color=sliceColors[(2*i)+1],lw=2,ls=":")
            ax1residuals.plot(xs_to_plot*np.sqrt(2), dataArray[startIdx:] - model_to_plot,color=sliceColors[(2*i)+1],lw=1)
            
        except np.linalg.LinAlgError:
            print(dataArray)

    #ax1residuals.set_ylabel("residuals",fontsize=18)
    ax1residuals.tick_params(axis='y',which='major',labelsize=16)
    ax1residuals.set_ylim(-3.,3.)
    ax1residuals.set_yticks(np.linspace(-3.,3.,3))
    #axResiduals.plot(xs_to_plot*np.sqrt(2),np.zeros_like(xs_to_plot),color='k',ls=':')
    ax1residuals.axhline(0,color='k',linestyle=":")
    ax1residuals.set_xticklabels([])
    #axResiduals.set_yticks([-2,-1,0,1,2])

    ax2.errorbar(rs, annFluxes,yerr=annFluxes_unc, capsize=0, color='k', elinewidth=0.5)
    ax2.set_ylabel("count rate [DN/s]"+"\n"+"summed over annulus",fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_yscale("log")



    peakIdx = np.argmax(annFluxes)

    rs_beyond = np.arange(0,int(512*np.sqrt(2)))

    startIdx = int(peakIdx + 0.5*(511 - peakIdx))
    endIdx = int(peakIdx + 0.7*(511 - peakIdx))
    print(startIdx,endIdx)

    window_to_fit = (rs >= startIdx) & (rs < endIdx)
    window_to_plot = (rs_beyond >= startIdx)

    rs_fit = rs[window_to_fit]
    rs_fit_scaled = (rs_fit - (startIdx-1))/(endIdx-startIdx) 

    rs_to_plot = rs_beyond[window_to_plot]
    rs_to_plot_scaled = (rs_to_plot - (startIdx-1))/(endIdx-startIdx) 

    best_params_exp = normal_equation(X=rs_fit_scaled,
                                      order=2,
                                      Y=np.log(annFluxes[window_to_fit]),
                                      Yerr=np.log(annFluxes_unc[window_to_fit]))
            
    model_to_plot = np.exp(best_params_exp[1])*np.exp(rs_to_plot_scaled)**best_params_exp[0]
    model_to_fit =  np.exp(best_params_exp[1])*np.exp(rs_fit_scaled)**best_params_exp[0]
    
    ax2.plot(rs_to_plot, model_to_plot,color=sliceColors[(2*i)+1],lw=2,ls=":")
    ax2residuals.plot(rs[startIdx:endIdx], annFluxes[startIdx:endIdx] - model_to_fit,color=sliceColors[(2*i)+1],lw=1)
    ax2residuals.axhline(0,color='k',ls=":")
    ax2residuals.set_xlabel("distance from center [pixels]",fontsize=18)
    
    for ax in [ax1,ax1residuals,ax2]:
        ax.set_xticklabels([])

    for ax in [ax1,ax1residuals,ax2,ax2residuals]:
        ax.axvline(512,color='k',ls='-')
        ax.set_xlim(-1,512*np.sqrt(2))
    
    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    if save is True:
        plt.savefig("{0}".format(saveFileName),bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return



def plot_other_EIT(fitsFileName, header, data, save=False, saveFileName=None):
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
        
    cs = ax0.imshow(np.log10(data),cmap='Greys_r',interpolation='None',origin="lower")
    ax0.set_title('{0}, {1}'.format(header['SCI_OBJ'], fitsFileName))
    
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad="2%")
    fig.add_axes(cax)
    fig.colorbar(cs, cax=cax,label="log_10(count)")

    if save is True:
        plt.savefig("{0}".format(saveFileName),bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return 



def make_disk_model(sidelength=1024,xcenter=512,ycenter=512,radius=953.884775291/2.627):
    """
    make a square array of shape (sidelength, sidelength) where 
    all pixels within "radius" of (xcenter,ycenter) have value 1
    all other pixels have value 0
    """
    square = np.zeros((sidelength,sidelength))
    square_idxs = np.indices((sidelength,sidelength))

    disk_mask = ( (square_idxs[0,:,:] - ycenter)**2 + (square_idxs[1,:,:] - xcenter)**2 <= radius**2)

    square[disk_mask] = 1.
    
    return square/np.sum(square)


def make_disk_corona_model(sidelength=1024,xcenter=512,ycenter=512,radius=953.884775291/2.627):
    """
    make a square array of shape (sidelength, sidelength) where 
    all pixels within "radius" of (xcenter,ycenter) have value 1
    all other pixels have value 0
    """
    square = np.zeros((sidelength,sidelength))
    square_idxs = np.indices((sidelength,sidelength))

    disk_mask = ( (square_idxs[0,:,:] - ycenter)**2 + (square_idxs[1,:,:] - xcenter)**2 < (0.95*radius)**2 )

    square[disk_mask] = 1.

    corona_mask =( ((square_idxs[0,:,:] - ycenter)**2 + (square_idxs[1,:,:] - xcenter)**2 >= (0.95*radius)**2)
                  & ((square_idxs[0,:,:] - ycenter)**2 + (square_idxs[1,:,:] - xcenter)**2 <= (1.05*radius)**2) )

    square[corona_mask] = np.sqrt(10.)
    
    return square/np.sum(square)

def chi_squared(data, model):
    """
    no uncertainties
    """
    return np.sum( np.abs(data - model) )


def fit_solar_center(data, header, guess_xcenter, guess_ycenter, search_box_radius, search_box_scale=1, model="disk+corona"):
    """
    fit a very approximate disk + corona model to find the indices of the pixel at the center of the solar disk
    """

    solRadPixels = header['RSUN_ARC']/header['CDELT1']

    normData = data/np.sum(data)

    if model == "disk+corona":
        bestChiSquared = chi_squared(normData, make_disk_corona_model(sidelength=1024,xcenter=guess_xcenter,ycenter=guess_ycenter,radius=solRadPixels))

    elif model == "disk":
        bestChiSquared = chi_squared(normData, make_disk_model(sidelength=1024,xcenter=guess_xcenter,ycenter=guess_ycenter,radius=solRadPixels))

    #coarse grid search
    for i in range(-search_box_radius, search_box_radius+1, search_box_scale):
        for j in range(-search_box_radius, search_box_radius+1, search_box_scale):
            if model == "disk+corona":
                chiSquared = chi_squared(normData, make_disk_corona_model(sidelength=1024,xcenter=guess_xcenter+j,ycenter=guess_ycenter+i,radius=solRadPixels))
            elif model == "disk":
                chiSquared = chi_squared(normData, make_disk_model(sidelength=1024,xcenter=guess_xcenter+j,ycenter=guess_ycenter+i,radius=solRadPixels))
               
            if chiSquared <= bestChiSquared:
                bestChiSquared = chiSquared
                xcenter = guess_xcenter+j
                ycenter = guess_ycenter+i

    #fine tune if necessary
    if search_box_scale > 1:
        guess_xcenter = xcenter
        guess_ycenter = ycenter
        
        for i in range(-search_box_scale, +search_box_scale+1):
            for j in range(-search_box_scale, search_box_scale+1):
                if model == "disk+corona":
                    chiSquared = chi_squared(normData, make_disk_corona_model(sidelength=1024,xcenter=guess_xcenter+j,ycenter=guess_ycenter+i,radius=solRadPixels))
                elif model == "disk":
                    chiSquared = chi_squared(normData, make_disk_model(sidelength=1024,xcenter=guess_xcenter+j,ycenter=guess_ycenter+i,radius=solRadPixels))
                   
                if chiSquared <= bestChiSquared:
                    bestChiSquared = chiSquared
                    xcenter = guess_xcenter+j
                    ycenter = guess_ycenter+i

    return xcenter, ycenter
    

def find_solar_center(data):
    """
    find the indices of the pixel at the center of the sun. this doesn't work if the whole disk isn't in the image!
    """
    sumOverColumns = np.sum(data,1)
    sumOverRows = np.sum(data,0)

    x_convolution = np.convolve(sumOverRows, sumOverRows)
    y_convolution = np.convolve(sumOverColumns, sumOverColumns)

    center_x = int(np.ceil(np.argmax(x_convolution)/2))
    center_y = int(np.ceil(np.argmax(y_convolution)/2))
    '''
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(np.arange(1024),sumOverColumns,'k-')
    ax.plot(np.arange(1024),sumOverRows,'r-')
    #ax.plot(x_convolution, 'k-')
    #ax.plot(y_convolution, 'r-')
    ax.axvline(center_x, color='r')
    ax.axvline(center_y, color='k')
    plt.show()
    '''
    return center_x, center_y


def findCoL(data):
    """
    find the indices of the pixel at the center of light in the image
    if the disk overlaps the edge of the image, this will not be the same as the center of the disk!
    """

    #center of light x, y coordinates

    #sum over columns = total light in each row
    sumOverColumns = np.sum(data,1)
    sumOverRows = np.sum(data,0)

    CoL_x = np.average(np.arange(np.shape(data)[1]), weights=sumOverRows)
    CoL_y = np.average(np.arange(np.shape(data)[0]), weights=sumOverColumns)

    return CoL_x, CoL_y
    

def center_from_header(header):
    """
    Get the central pixel from the information in the FITS header
    """

    # spacecraft roll angle
    rot = header['CROTA'] * (np.pi/180) # convert to radians
    rot_mat = np.array(((np.cos(rot), np.sin(rot)),(-np.sin(rot), np.cos(rot))))

    # header 'CRVAL1' and 'CRVAL2' are in world coordinates, algined with solar equator/poles
    xy_wcs = np.atleast_2d(np.array((header['CRVAL1'], header['CRVAL2']))).T

    # convert back to image coordinates
    xy_img = np.matmul(rot_mat, xy_wcs)

    xcenter = header['CRPIX1'] - xy_img[0][0]/header['CDELT1']
    ycenter = header['CRPIX2'] - xy_img[1][0]/header['CDELT2']

    #xcenter = header['CRPIX1'] - header['CRVAL1']/header['CDELT1']
    #ycenter = header['CRPIX2'] - header['CRVAL2']/header['CDELT2']

    return xcenter, ycenter

