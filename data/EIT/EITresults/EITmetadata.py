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

#World Coordinate System Attitude section of header
list_of_wcsa_keywords = ['WCSNAME','CTYPE1','CTYPE2','CUNIT1','CUNIT2','PC1_1','PC1_2','PC2_1','PC2_2','CDELT1','CDELT2','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CROTA']

#Temperature section of header
list_of_temp_keywords = ['CFTEMP','CCDTEMP']

#Solar Ephemeris section of header
list_of_se_keywords = ['SOLAR_B0','RSUN_ARC','RSUN_OBS','RSUN_REF','CAR_ROT','DSUN_OBS','SC_X0','SC_Y0','SC_ROLL','HAEX_OBS','HAEY_OBS','HAEZ_OBS']

#other metadata 
list_of_other_keywords = ['DATE-BEG','DATE-AVG','XPOSURE','CMDXPOS','SHUTCLS','FILTER','WAVELNTH','OBJECT','SCI_OBJ','OBS_PROG','CMP_NO','UCD','EXPMODE','LYNESYNC','ROUTPORT','NLEBPROC','LEBPROC1','LEBPROC2','LEBPROC3']


n_cols = len(list_of_other_keywords) + len(list_of_wcsa_keywords) + len(list_of_temp_keywords) + len(list_of_se_keywords)

toSave = np.zeros((1,n_cols))

toSaveHeader = list_of_other_keywords + list_of_wcsa_keywords + list_of_temp_keywords + list_of_se_keywords
toSaveHeaderStr = ' '.join(toSaveHeader)
print(toSaveHeaderStr)
toSaveFormat = "%f %f %f %f %f %s %d %s %s %s %f %s %s %s %s %d %d %d %d" + "%s %s %s %s %s %f %f %f %f %f %f %f %f %f %f %f " + "%f %f " + "%f %f %f %f %f %f %f %f %f %f %f %f"

print(np.shape(toSave))
print(np.shape(toSaveHeader))
for year in range(1996, 2023):
#for year in range(2011,2012):
    jan1_thisyear = '{0}-01-01T00:00:00.000Z'.format(year)
    jan1_thisyear = astropy.time.Time(jan1_thisyear).jd
    print(year)

    for m in range(1,13):
        month = str(m).zfill(2)
        #catch missing months (i.e. July-September 1998)
        try:
            days = sorted([int(f.path[-2:]) for f in os.scandir("./{0}/{1}/".format(year,month)) if f.is_dir()])
            #days = sorted([int(f.path[-2:]) for f in os.scandir("../EIT/{0}/{1}/".format(year,month)) if f.is_dir()])
        except FileNotFoundError:
            continue
        print(month)
        
        for d in days:
            day = str(d).zfill(2)
            fitsFiles = os.scandir("./{0}/{1}/{2}/".format(year,month,day))
            #fitsFiles = os.scandir("../EIT/{0}/{1}/{2}/".format(year,month,day))
            print(d)
            
            for f in fitsFiles:
                hdul = fits.open(f.path)
                header = hdul[0].header
                #data = hdul[0].data

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
                    if header['NAXIS1'] == 1024 and header['NAXIS2'] == 1024 and header['MSBLOCKS'] == 0:
                        
                        if header['NLEBPROC'] != 0:
                            print(header['NLEBPROC'])

                        t_beg = astropy.time.Time(header['DATE-BEG'], format='isot', scale='utc').jd
                        t_avg = astropy.time.Time(header['DATE-AVG'], format='isot', scale='utc').jd

                        toSave_thisImg = [t_beg, t_avg]

                        for i,k in enumerate(list_of_other_keywords[2:]):
                            try:
                                if isinstance(header[k],str):
                                    entry_toSave = header[k].replace(" ","_")
                                else:
                                    entry_toSave = header[k]
                                toSave_thisImg.append(entry_toSave)
                            except KeyError:
                                toSave_thisImg.append(np.nan)
                        
                        for i,k in enumerate(list_of_wcsa_keywords):
                        	toSave_thisImg.append(header[k])

                        for i,k in enumerate(list_of_temp_keywords):
                        	toSave_thisImg.append(header[k])

                        for i,k in enumerate(list_of_se_keywords):
                        	toSave_thisImg.append(header[k])
                        
                        toSave = np.vstack((toSave, np.array(toSave_thisImg)))

print(np.shape(toSave))
print(toSave[0])
toSave = toSave[1:]
print(np.shape(toSave))
print(toSave[0])
np.savetxt("./EIT_headerMetadata.txt",toSave,fmt='%s', header=toSaveHeaderStr)



