import numpy as np
import os

from astropy.io import ascii, fits
from astropy.table import Table
import astropy.time

dark_times = []
dark_medians = []
dark_means = []
dark_stdevs = []
    
for year in range(1996, 2023):
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
            
            for f in fitsFiles:
                if "DRK" in f.path:
                    hdul = fits.open(f.path)
                    header = hdul[0].header
                    data = hdul[0].data
                
                    sci_obj_ = header['SCI_OBJ'].replace("/"," ").replace(" ","_").lower()

                    #exclude pathological case where solar distance is 0
                    if header['DSUN_OBS'] < 1.e11:
                        continue
                    
                    if "dark_image" in sci_obj_:
                        #separate out what I assume are mis-labeled stray light images
                        if np.median(data) < 1000.:
                            dark_times.append(header['DATE-BEG']) 
                            dark_medians.append(np.median(data))
                            dark_means.append(np.mean(data))
                            dark_stdevs.append(np.std(data))

                    hdul.close()
                
dark_times = astropy.time.Time(dark_times, format='isot', scale='utc').jd
dark_medians = np.array(dark_medians)
dark_means = np.array(dark_means)
dark_stdevs = np.array(dark_stdevs)

dark_medians = dark_medians[np.argsort(dark_times)]
dark_means = dark_means[np.argsort(dark_times)]
dark_stdevs = dark_stdevs[np.argsort(dark_times)]
dark_times = dark_times[np.argsort(dark_times)]

print(np.shape(dark_times))
print(np.shape(dark_medians))
print(np.shape(dark_means))
print(np.shape(dark_stdevs))


dark_toSave = np.vstack((dark_times.T,dark_means.T,dark_stdevs.T)).T
print(np.shape(dark_toSave))

np.savetxt("./darkImgFluxes.txt",dark_toSave,fmt="%f", header="t_obs[JD] flux_mean[DN/s] flux_stdev[DN/s]")





