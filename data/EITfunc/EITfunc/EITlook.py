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

import scipy.optimize as sciop
from scipy.stats import mode, binned_statistic

import time

__all__ = ['draw_solar_rot', 'plot_full_sun_EIT', 'plot_other_EIT', 'make_disk_model', 'make_disk_corona_model','chi_squared','fit_solar_center','find_solar_center','findCoL','center_from_header']


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

    xcenter = header['CRPIX1'] - header['CRVAL1']/header['CDELT1']
    ycenter = header['CRPIX2'] - header['CRVAL2']/header['CDELT2']

    return xcenter, ycenter

