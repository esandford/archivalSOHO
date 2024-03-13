import numpy as np

import os

from astropy import units as u
from astropy.constants import G
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.timeseries import LombScargle
import astropy.time

import scipy.optimize as sciop
from scipy.stats import mode, binned_statistic


#EIT functions
def interpolate_darkcurrent(obsTime, darkFramesArr):
    dark_times = darkFramesArr[:,0]
    dark_fluxes = darkFramesArr[:,1]
    dark_uncs = darkFramesArr[:,2]

    i_prev = np.arange(len(darkFramesArr))[dark_times < obsTime][-1]
    i_next = i_prev + 1

    interp_dark_flux = dark_fluxes[i_prev] + ((dark_fluxes[i_next] - dark_fluxes[i_prev])*((obsTime - dark_times[i_prev])/(dark_times[i_next]-dark_times[i_prev])))

    return interp_dark_flux


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
        annFluxes_unc[i] = np.sqrt(annFlux + (annNpix * darkFlux))

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


def center_from_header(header):
    """
    Get the central pixel from the information in the FITS header
    """

    xcenter = header['CRPIX1'] - header['CRVAL1']/header['CDELT1']
    ycenter = header['CRPIX2'] - header['CRVAL2']/header['CDELT2']

    return xcenter, ycenter



darkImgData =np.loadtxt("./darkImgFluxes_nobakeout.txt")
print(np.shape(darkImgData))

ts = []
ds = []
fs = []
us = []
ws = []

#for year in range(1996, 2010):
for year in [1996]:
    jan1_thisyear = '{0}-01-01T00:00:00.000Z'.format(year)
    jan1_thisyear = astropy.time.Time(jan1_thisyear).jd
    print(year)

    for m in range(1,13):
        month = str(m).zfill(2)
        #catch missing months (i.e. July-September 1998)
        try:
            days = sorted([int(f.path[-2:]) for f in os.scandir("./{0}/{1}/".format(year,month)) if f.is_dir()])
        except FileNotFoundError:
            continue
        print(month)
        
        for d in days:
            day = str(d).zfill(2)
            fitsFiles = os.scandir("./{0}/{1}/{2}/".format(year,month,day))
            print(d)
            
            for f in fitsFiles:
                hdul = fits.open(f.path)
                header = hdul[0].header
                data = hdul[0].data

                sci_obj_ = header['SCI_OBJ'].replace("/"," ").replace(" ","_").lower()
                dSun = header['DSUN_OBS']
            
                #exclude pathological case where solar distance is 0
                if dSun < 1.e11:
                    continue

                #find center of solar disk
                xcenter, ycenter = center_from_header(header) # pixels

                if "full_sun" in sci_obj_: 
                    hist = str(header['HISTORY'])

                    # catch the one image with sci_obj_ = 'full_sun_304' which
                    # doesn't have a wavelength keyword (it seems to be a mislabeled dark image?) 
                    if "Data not calibrated" in hist:
                        continue

                    #exclude lower-resolution or partial images and images with missing data blocks
                    if np.shape(data) == (1024,1024) and header['MSBLOCKS'] == 0 and header['CAMERERR'] == 'no':
                        
                        t = astropy.time.Time(header['DATE-BEG'], format='isot', scale='utc').jd
                        
                        if t > darkImgData[0][0] and t < darkImgData[-1][0]:
                            dark = interpolate_darkcurrent(t, darkImgData)
                        elif t < darkImgData[0][0]:
                            dark = darkImgData[0][1]
                        elif t > darkImgData[-1][0]:
                            dark = darkImgData[-1][1]

                        f, u = image_to_LCpoint(data, darkFlux=dark, xcenter=xcenter, ycenter=ycenter, maskBrightSpots=True)
                        w = header['WAVELNTH']
                
                        ts.append(t)
                        ds.append(dSun)
                        fs.append(f)#*(medianSolarDistance/dSun))
                        us.append(u)#*(medianSolarDistance/dSun))
                        ws.append(w)
                                 
ts = np.array(ts)
ds = np.array(ds)
fs = np.array(fs)
us = np.array(us)
ws = np.array(ws)

ds = ds[np.argsort(ts)]
fs = fs[np.argsort(ts)]
us = us[np.argsort(ts)]
ws = ws[np.argsort(ts)]
ts = ts[np.argsort(ts)]

print(np.shape(ts)) # =2877 in 1996, =2520 in 2011
print(np.shape(ds))
print(np.shape(fs))
print(np.shape(us))
print(np.shape(ws))

toSave = np.vstack((ts.T,ds.T,ws.T,fs.T,us.T)).T
print(np.shape(toSave))

np.savetxt("./EIT_LC.txt",toSave,fmt="%f", header="t_obs[JD] solar_distance[m] wavelength[angstrom] unnorm_flux[DN/s] unnorm_flux_unc[DN/s]")



