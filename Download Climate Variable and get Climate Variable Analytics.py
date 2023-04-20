# %% [markdown]
# DOWNLOAD

# %%
%reload_ext autoreload
%autoreload 2
%reload_ext watermark
%matplotlib inline
%config InlineBackend.figure_format = "retina"

import xarray as xr
import intake
import regionmask

from matplotlib import rcParams

import cartopy.crs as ccrs
from carbonplan import styles  # noqa: F401

import warnings

regionmask.__version__
warnings.filterwarnings("ignore")

# Note: cmip6_downscaling repo requires PYTHON VERSION 3.8 (or greater) and 

from cmip6_downscaling.analysis.analysis import (
    grab_big_city_data,
    load_big_cities,
)

from cmip6_downscaling.analysis.plot import plot_city_data
xr.set_options(keep_attrs=True)

# %%
cat = intake.open_esm_datastore(
    "https://cpdataeuwest.blob.core.windows.net/cp-cmip/version1/catalogs/global-downscaled-cmip6.json"
)

# %%
# # ------ MIN TEMP SUBSET ------
cat_subset_min = cat.search(
    method="GARD-SV",
    source_id="MRI-ESM2-0",
    experiment_id="ssp245",
    variable_id="tasmin",
    timescale="month"
)
cat_subset_min.df.head()

# %%
# ------ MAX TEMP SUBSET ------
cat_subset_max = cat.search(
    method="GARD-SV",
    source_id="MRI-ESM2-0",
    experiment_id="ssp245",
    variable_id="tasmax",
    timescale="month"
)
cat_subset_max.df.head()

# %%
# ------ PRECIPITATION SUBSET ------
cat_subset_pr = cat.search(
    method="GARD-SV",
    source_id="MRI-ESM2-0",
    experiment_id="ssp245",
    variable_id="pr",
    timescale="month"
)
cat_subset_pr.df.head()

# %%
dsets_min = cat_subset_min.to_dataset_dict()
dsets_min

# %%
dsets_max = cat_subset_max.to_dataset_dict()
dsets_max

# %%
dsets_pr = cat_subset_pr.to_dataset_dict()
dsets_pr

# %%
ds_min = dsets_min["ScenarioMIP.MRI.MRI-ESM2-0.ssp245.month.GARD-SV"]
ds_min

# %%
ds_max = dsets_max["ScenarioMIP.MRI.MRI-ESM2-0.ssp245.month.GARD-SV"]
ds_max

# %%
ds_pr = dsets_pr["ScenarioMIP.MRI.MRI-ESM2-0.ssp245.month.GARD-SV"] 
ds_pr

# %%
ds_min -= 273.15
ds_max -= 273.15

# %%
land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
projection = ccrs.PlateCarree()

# %%
## --- SET REGION ---
##  India: lat(8,37) , lon(68,97) (Note: it takes around 20 minutes to run for a single year over India)
##  Bihar: lat(24,28), lon(83,89)
##  Madhya Pradesh: lat(21,27), lon(73,83)

# Region
region = {"lat": slice(24,28), "lon": slice(83,89)} 

# %%
# DOWNLOAD MIN TEMP
tasmin = ds_min.tasmin.sel(time=slice("2020-1-1", "2099")).sel(**region).load() #xarray dataarray

# %%
tasmin

# %%
# SAVE MIN TEMP DATA AS NETCDF
path="/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Min_Temp_2020-2099.nc"
tasmin.to_netcdf(path, mode='w', format="NETCDF4", group=None, engine="netcdf4", encoding=None, unlimited_dims=None, compute=True, invalid_netcdf=False)

# %%
# DOWNLOAD MAX TEMP
tasmax = ds_max.tasmax.sel(time=slice("2020", "2099")).sel(**region).load() #xarray dataarray

# %%
tasmax

# %%
# SAVE MAX TEMP DATA AS NETCDF
path="/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Max_Temp_2020-2099.nc"
tasmax.to_netcdf(path, mode='w', format="NETCDF4", group=None, engine="netcdf4", encoding=None, unlimited_dims=None, compute=True, invalid_netcdf=False)

# %%
# # DOWNLOAD PRECIPITATION
precip = ds_pr.pr.sel(time=slice("2020", "2099")).sel(**region).load() #xarray dataarray

# %%
# SAVE PRECIPITATION DATA AS NETCDF
path="/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Precip_2020-2099.nc"
precip.to_netcdf(path, mode='w', format="NETCDF4", group=None, engine="netcdf4", encoding=None, unlimited_dims=None, compute=True, invalid_netcdf=False)

# %% [markdown]
# CLIP

# %%
# IMPORT LIBRARIES
import os, sys 
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import datetime
from osgeo import gdal

%matplotlib inline

# %%
# FETCH NETCDF FILES
f_min = "/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Min_Temp_2020-2099.nc"
f_max = "/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Max_Temp_2020-2099.nc"
f_pr = "/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Precip_2020-2099.nc"

rec_min = Dataset("/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Min_Temp_2020-2099.nc")
rec_max = Dataset("/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Max_Temp_2020-2099.nc")
rec_pr = Dataset("/Users/revatishelat/Documents/DownIES/Data/Bihar/Rectangular/Rec_Monthly_Precip_2020-2099.nc")

# %%
print("\n-----Minimum Temperature-----\n")
tmin = rec_min.variables['tasmin']
print(tmin)

# %%
print("\n-----Maximum Temperature-----\n")
tmax = rec_max.variables['tasmax']
print(tmax)

# %%
print("\n-----Precipitation-----\n")
tpr = rec_pr.variables['pr']
print(tpr)

# %%
print("Units\n----------------------------")
unitmin = rec_min.variables['tasmin'].units
print("Temperature: "+unitmin)

unitpr = rec_pr.variables['pr'].units
print("Precipitation: "+unitpr)

unitlat = rec_min.variables['lat'].units
print("Latitude: " +unitlat)

unitlon = rec_min.variables['lon'].units
print("Longitude: " +unitlon)

unitt = rec_min.variables['time'].units
print("Time: "+unitt)

# %%
# (ARRAY OF) MINIMUM TEMPERATURE VALUES

tmin = rec_min.variables['tasmin'][0,:,:,:].data #precipitation
print(tmin)
print(len(tmin))

# %%
tmax = rec_max.variables['tasmax'][0:,:,:].data # 0 indicates first time slice
print(tmax)
print(len(tmax))

# %%
# (ARRAY OF) MINIMUM TEMPERATURE VALUES

tpr = rec_pr.variables['pr'][0,:,:,:].data #precipitation
print(tpr)
print(len(tpr))

# %%
import xarray as xr # for reading netcdf dataset
import numpy as np 
import geopandas as gpd #dealing with shape file
import regionmask 
import pandas as pd 
import cartopy.crs as ccrs #plotting
import matplotlib.pyplot as plt 
import warnings; warnings.filterwarnings(action='ignore')
import time

%matplotlib inline

# %%
shapefile = "/Users/revatishelat/Documents/DownIES/India and States shapefiles/bihar.shp"

state = gpd.read_file(shapefile)
state

# %%
state.geometry

# %%
fig, ax = plt.subplots(figsize=(25,17))
state.plot(ax=ax, column = "stname")

my_list = list(state['stname'])
my_list_unique = set(my_list)
indexes = [my_list.index(x) for x in my_list_unique]
print(my_list_unique)

# %%
state.geometry

# %%
# CREATE REGION

state_mask_poly = regionmask.Regions(name=state.stname, numbers=indexes, names=state.stname[indexes],outlines=state.geometry)
state_mask_poly

# %%
print("{}".format(state_mask_poly.names[:]))

# %%
# READ NETCDF FILE AS XARRAY
a_min = xr.open_dataarray(f_min)
a_min

# %%
a_max  = xr.open_dataarray(f_max)
a_max

# %%
a_pr = xr.open_dataarray(f_pr)
a_pr

# %%
# CREATING MASK

mask = state_mask_poly.mask(a_min.isel(time =0), lat_name='lat', lon_name='lon') #time = 0 => one time slice; this takes only one latitude and longitude #.isel(time =0)
mask

# %%
f_mask = "/Users/revatishelat/Documents/DownIES/mask_by_Bihar.nc"
mask.to_netcdf(f_mask)

# %%
#Read mask as saved previously

mask1 = xr.open_dataarray(f_mask)
print(mask1)
print(type(mask1))

# %%
masked_shape_min = a_min.where(mask1 == 0)
print(masked_shape_min)

# %%
masked_shape_max = a_max.where(mask == 0)
masked_shape_max

# %%
masked_shape_pr = a_pr.where(mask == 0)
masked_shape_pr

# %%
#Plot masked data (one we've obtained above)

plt.figure(figsize=(25, 17))
ax = plt.axes()
masked_shape_min.isel(time=0).plot(ax=ax)
state.plot(ax=ax, alpha = 0.8, facecolor = 'none')

# %%
#Plot masked data (one we've obtained above)

plt.figure(figsize=(25, 17))
ax = plt.axes()
masked_shape_max.isel(time=0).plot(ax=ax)
state.plot(ax=ax, alpha = 0.8, facecolor = 'none')

# %%
#Plot masked data (one we've obtained above)

plt.figure(figsize=(25, 17))
ax = plt.axes()
masked_shape_pr.isel(time=0).plot(ax=ax)
state.plot(ax=ax, alpha = 0.8, facecolor = 'none')

# %%
encoding = {"tasmin": {'zlib':True, "complevel":4}}
masked_shape_min.to_netcdf("/Users/revatishelat/Documents/DownIES/Data/Bihar/Clipped/Clipped_Monthly_Min_Temp_2020-2099.nc", format='NETCDF4',engine='netcdf4',encoding=encoding)

# %%
encoding = {"tasmax": {'zlib':True, "complevel":4}}
masked_shape_max.to_netcdf("/Users/revatishelat/Documents/DownIES/Data/Bihar/Clipped/Clipped_Monthly_Max_Temp_2020-2099.nc", format='NETCDF4',engine='netcdf4', encoding=encoding)

# %%
encoding = {"pr": {'zlib':True, "complevel":4}}
masked_shape_pr.to_netcdf("/Users/revatishelat/Documents/DownIES/Data/Bihar/Clipped/Clipped_Monthly_Precip_2020-2099.nc", format='NETCDF4',engine='netcdf4', encoding=encoding)

# %% [markdown]
# VISUALIZE

# %%
# IMPORT LIBRARIES
import netCDF4 as nc
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from pandas import DataFrame
import seaborn as sb
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)

# %%
## ---- CLIPPED DATA FILES ----
f_tasmin = "/Users/revatishelat/Documents/DownIES/Data/Bihar/Clipped/Clipped_Monthly_Min_Temp_2020-2099.nc"
f_tasmax = "/Users/revatishelat/Documents/DownIES/Data/Bihar/Clipped/Clipped_Monthly_Max_Temp_2020-2099.nc"
f_pr = "/Users/revatishelat/Documents/DownIES/Data/Bihar/Clipped/Clipped_Monthly_Precip_2020-2099.nc"

# %%
ds_tasmin = Dataset(f_tasmin)
print(ds_tasmin)

# %%
ds_tasmax = Dataset(f_tasmax)
print(ds_tasmax)

# %%
ds_pr = Dataset(f_pr)
print(ds_pr)

# %%
# Variables
print(ds_tasmin.variables.keys())
print(ds_tasmax.variables.keys())
print(ds_pr.variables.keys())

# %%
no_of_months = ds_tasmin.variables['time'].size
no_of_months

# %%
years = list(range(2020,2100))
print(len(years))
print(years)

# %%
# Monthly Minimum Temperature Averages
means_min = []
for i in range(0,no_of_months):
        a = ds_tasmin['tasmin'][0,i].data
        a = a[np.logical_not(np.isnan(a))]
        j = np.ndarray.mean(a)
        means_min.append(j)
print(means_min)

# %%
# Monthly Maximum Temperature Averages
means_max = []
for i in range(0,no_of_months):
        a = ds_tasmax['tasmax'][0,i].data
        a = a[np.logical_not(np.isnan(a))]
        j = np.ndarray.mean(a)
        means_max.append(j)
print(means_max)

# %%
# Monthly Precipitation Averages
means_pr = []
for i in range(0,no_of_months):
        a = ds_pr['pr'][0,i].data
        a = a[np.logical_not(np.isnan(a))]
        j = np.ndarray.mean(a)
        means_pr.append(j)
print(means_pr)

# %%
print(len(means_min))
print(len(means_max))
print(len(means_pr))

# %%
# Divide data into sublists of years
def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
 
# How many elements each list should have
n = 12

x_min = list(divide_chunks(means_min, n))
print (x_min)

# %%
# Divide data into sublists of years
def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
 
# How many elements each list should have
n = 12

x_max = list(divide_chunks(means_max, n))
print (x_max)

# %%
# Divide data into sublists of years
def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
 
# How many elements each list should have
n = 12
 
x_pr = list(divide_chunks(means_pr, n))
print (x_pr)

# %% [markdown]
# SEASONAL AVERAGES

# %%
# Avg Min Temp

yrly_avg_mins_jf =[]
yrly_avg_mins_mam = []
yrly_avg_mins_jjas = []
yrly_avg_mins_ond = []
for i in range(0,len(years)):
    y=x_min[i]
    # jan, feb
    j = ((y[0]+y[1])/2)
    yrly_avg_mins_jf.append(j)

    # mar, apr, may
    k = ((y[2]+y[3]+y[4])/3)
    yrly_avg_mins_mam.append(k)

    # jun, jul, aug, sept
    l = ((y[5]+y[6]+y[7]+y[8])/4)
    yrly_avg_mins_jjas.append(l)

    # oct, nov, dec
    m = (y[9]+y[10]+y[11])/3
    yrly_avg_mins_ond.append(m)

print(yrly_avg_mins_jf)
print(yrly_avg_mins_mam)
print(yrly_avg_mins_jjas)
print(yrly_avg_mins_ond)

# %%
yrly_avg_mins_jf = pd.DataFrame(yrly_avg_mins_jf)
yrly_avg_mins_mam = pd.DataFrame(yrly_avg_mins_mam)
yrly_avg_mins_jjas = pd.DataFrame(yrly_avg_mins_jjas)
yrly_avg_mins_ond = pd.DataFrame(yrly_avg_mins_ond)

seasonal_mins = pd.concat([yrly_avg_mins_jf, yrly_avg_mins_mam, yrly_avg_mins_jjas, yrly_avg_mins_ond], axis="columns")
seasonal_mins.columns = ["Winter", "Summer", "Monsoon", "Autumn"]
seasonal_mins.index = [years]

seasonal_mins

# %%
# Avg Max Temp

yrly_avg_maxs_jf =[]
yrly_avg_maxs_mam = []
yrly_avg_maxs_jjas = []
yrly_avg_maxs_ond = []

for i in range(0,len(years)):
    y=x_max[i]
    # jan, feb
    j = ((y[0]+y[1])/2)
    yrly_avg_maxs_jf.append(j)

    # mar, apr, may
    k = ((y[2]+y[3]+y[4])/3)
    yrly_avg_maxs_mam.append(k)

    # jun, jul, aug, sept
    l = ((y[5]+y[6]+y[7]+y[8])/4)
    yrly_avg_maxs_jjas.append(l)

    # oct, nov, dec
    m = (y[9]+y[10]+y[11])/3
    yrly_avg_maxs_ond.append(m)

print(yrly_avg_maxs_jf)
print(yrly_avg_maxs_mam)
print(yrly_avg_maxs_jjas)
print(yrly_avg_maxs_ond)

# %%
yrly_avg_maxs_jf = pd.DataFrame(yrly_avg_maxs_jf)
yrly_avg_maxs_mam = pd.DataFrame(yrly_avg_maxs_mam)
yrly_avg_maxs_jjas = pd.DataFrame(yrly_avg_maxs_jjas)
yrly_avg_maxs_ond = pd.DataFrame(yrly_avg_maxs_ond)

seasonal_maxs = pd.concat([yrly_avg_maxs_jf, yrly_avg_maxs_mam, yrly_avg_mins_jjas, yrly_avg_maxs_ond], axis="columns")
seasonal_maxs.columns = ["Winter", "Summer", "Monsoon", "Autumn"]
seasonal_maxs.index = [years]

seasonal_maxs

# %%
# Avg Precipitation

yrly_avg_prs_jf =[]
yrly_avg_prs_mam = []
yrly_avg_prs_jjas = []
yrly_avg_prs_ond = []

for i in range(0,len(years)):
    y=x_pr[i]
    # jan, feb
    j = ((y[0]+y[1])/2)
    yrly_avg_prs_jf.append(j)

    # mar, apr, may
    k = ((y[2]+y[3]+y[4])/3)
    yrly_avg_prs_mam.append(k)

    # jun, jul, aug, sept
    l = ((y[5]+y[6]+y[7]+y[8])/4)
    yrly_avg_prs_jjas.append(l)

    # oct, nov, dec
    m = (y[9]+y[10]+y[11])/3
    yrly_avg_prs_ond.append(m)

print(yrly_avg_prs_jf)
print(yrly_avg_prs_mam)
print(yrly_avg_prs_jjas)
print(yrly_avg_prs_ond)

# %%
yrly_avg_prs_jf = pd.DataFrame(yrly_avg_prs_jf)
yrly_avg_prs_mam = pd.DataFrame(yrly_avg_prs_mam)
yrly_avg_prs_jjas = pd.DataFrame(yrly_avg_prs_jjas)
yrly_avg_prs_ond = pd.DataFrame(yrly_avg_prs_ond)

seasonal_prs = pd.concat([yrly_avg_prs_jf, yrly_avg_prs_mam, yrly_avg_prs_jjas, yrly_avg_prs_ond], axis="columns")
seasonal_prs.columns = ["Winter", "Summer", "Monsoon", "Autumn"]
seasonal_prs.index = years

seasonal_prs

# %% [markdown]
# Save to CSV

# %%
pd.DataFrame.transpose(seasonal_mins).to_csv("/Users/revatishelat/Documents/DownIES/Visualization/Seasonal/Seasonal_Average_Minimum_Temperature.csv",header=years,index_label="Seasons")

# %%
pd.DataFrame.transpose(seasonal_maxs).to_csv("/Users/revatishelat/Documents/DownIES/Visualization/Seasonal/Seasonal_Average_Maximum_Temperature.csv" ,header=years, index_label="Seasons")

# %%
pd.DataFrame.transpose(seasonal_prs).to_csv("/Users/revatishelat/Documents/DownIES/Visualization/Seasonal/Seasonal_Average_Precipitation.csv",header=years, index_label="Seasons")

# %%
from sklearn.linear_model import LinearRegression
from scipy import stats

# %%
# add years column
seasonal_mins.insert(4, 'Years', years)
seasonal_mins

# %%
# regression lines
slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_mins['Years'],seasonal_mins['Winter'])
ax = sb.regplot(x="Years", y="Winter", ci=None, data = seasonal_mins, line_kws={'label':"Winter y = {0:.3f}x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Winter): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 13))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_mins['Years'],seasonal_mins['Summer'])
ax = sb.regplot(x="Years", y="Summer", ci=None, data = seasonal_mins, line_kws={'label':"Summer y = {0:.3f}x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Summer): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 23))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_mins['Years'],seasonal_mins['Monsoon'])
ax = sb.regplot(x="Years", y="Monsoon", ci=None, data = seasonal_mins, line_kws={'label':"Monsoon y = {0:.3f}x+ ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Monsoon): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 24.5))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_mins['Years'],seasonal_mins['Autumn'])
ax = sb.regplot(x="Years", y="Autumn", ci=None, data = seasonal_mins, line_kws={'label':"Autumn y = {0:.3f}x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Autumn): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 17.5))

# plot legend
ax.legend()

# axes labels
plt.xlabel('Years', fontsize=12)
plt.ylabel('Average Minimum Temperature', fontsize=12)
plt.title("Average Minimum Temperature in Bihar (by season) from 2022 to 2099", fontsize=14)

# figure size
sb.set(rc={"figure.figsize":(14, 10)})

# save plot
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("/Users/revatishelat/Documents/DownIES/Visualization/Seasonal/Average Minimum Temperature in Bihar (by seasons) 2020-2099.")

# %%
# add years column
seasonal_maxs.insert(4, 'Years', years)
seasonal_maxs

# %%
# regression lines
slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_maxs['Years'],seasonal_maxs['Winter'])
ax = sb.regplot(x="Years", y="Winter", ci=None, data = seasonal_maxs, line_kws={'label':"Winter y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Winter): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 24))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_maxs['Years'],seasonal_maxs['Summer'])
ax = sb.regplot(x="Years", y="Summer", ci=None, data = seasonal_maxs, line_kws={'label':"Summer y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Summer): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 34.25))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_maxs['Years'],seasonal_maxs['Monsoon'])
ax = sb.regplot(x="Years", y="Monsoon", ci=None, data = seasonal_maxs, line_kws={'label':"Monsoon y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Monsoon): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 26))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_maxs['Years'],seasonal_maxs['Autumn'])
ax = sb.regplot(x="Years", y="Autumn", ci=None, data = seasonal_maxs, line_kws={'label':"Autumn y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Autumn): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 28))

# plot legend
ax.legend()

# axes labels
plt.xlabel('Years', fontsize=12)
plt.ylabel('Average Maximum Temperature', fontsize=12)
plt.title("Average Maximum Temperature in Bihar (by season) from 2022 to 2099", fontsize=14)

# figure size
sb.set(rc={"figure.figsize":(14, 10)})

# save plot
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("/Users/revatishelat/Documents/DownIES/Visualization/Seasonal/Average Maximum Temperature in Bihar (by seasons) 2020-2099.")

# %%
# add years column
seasonal_prs.insert(4, 'Years', years)
seasonal_prs

# %%
# regression lines
slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_prs['Years'],seasonal_prs['Winter'])
ax = sb.regplot(x="Years", y="Winter", ci=None, data = seasonal_prs, line_kws={'label':"Winter y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Winter): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 0))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_prs['Years'],seasonal_prs['Summer'])
ax = sb.regplot(x="Years", y="Summer", ci=None, data = seasonal_prs, line_kws={'label':"Summer y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Summer): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 60))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_prs['Years'],seasonal_prs['Monsoon'])
ax = sb.regplot(x="Years", y="Monsoon", ci=None, data = seasonal_prs, line_kws={'label':"Monsoon y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Monsoon): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 250))

slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_prs['Years'],seasonal_prs['Autumn'])
ax = sb.regplot(x="Years", y="Autumn", ci=None, data = seasonal_prs, line_kws={'label':"Autumn y = ({0:.3f})x + ({1:.3f})".format(slope,intercept)})
ax.annotate('R^2 (Autumn): ' + str("{:.3f}".format(r_value**2)), xy=(2020, 80))

# plot legend
ax.legend()

# axes labels
plt.xlabel('Years', fontsize=12)
plt.ylabel('Average Precipitation', fontsize=12)
plt.title("Average Rainfall in Bihar (by season) from 2022 to 2099", fontsize=14)

# figure size
sb.set(rc={"figure.figsize":(14, 10)})

# save plot
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("/Users/revatishelat/Documents/DownIES/Visualization/Seasonal/Average Precipitation in Bihar (by seasons) 2020-2099.")


