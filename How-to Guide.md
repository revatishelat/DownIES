### DownIES How-to Guide

*In your conda environment, please use python version 3.8!*

---
**1. To customize dataset:**<br>
(subsets in DownIES Jupyter Notebook are for Minimum temperature, Maximum temperature, Precipitation from <u>MRI-ESM2-0</u> **GCM**, using <u>GARD-SV</u> **downscaling algorithm** on monthly timescale for the <u>SSP2</u> **emission scenario**)<br>

**Choose from:** <br>
<u>Downscaling algorithms (method) </u> <br>
DeepSD-BC<br>
DeepSD<br>
GARD-MV<br>
GARD-SV<br>
MACA<br>

<u>GCMs (source_id) </u> <br>
CanESM5 <br>
MRI-ESM2-0<br>
NorESM2-LM<br>
BCC-CSM2-MR<br>
MIROC6<br>
MPI-ESM1-2-HR<br>

<u>Scenarios (experimental_id)</u> <br>
historical (Historical)<br>
ssp245 (SSP2 4.5)<br>
ssp370 (SSP3 7.0)<br>
ssp585 (SSP5 8.5)<br>

<u>Climate variables (variable_id)</u> <br>
tasmin (Min temp)<br>
tasmax (Max temp)<br>
pr (Precipitation)<br>
<br>

<u>Time scale (timescale)</u> <br>
year<br>
month<br>
day<br>
<br>

---
**2. To download dataset**<br>
To download data, set latitude and longitude for the desired area.

For example, in our project, the latitude and longitude ranges for bihar are 24-28 and 83-89. In this chuck, the ranges can be replaced for the desired area. <br>

*Set region*
```
region = {"lat": slice(24,28), "lon": slice(83,89)} 
```
*Select time* <br>
Select a time from between 2015 and 2099; it follows syntax yyyy-mm-dd. <br>
(Note: for monthly data, the day remains 1 (eg. yyyy-mm-1); for yearly data, the day and month remain 1 (yyyy-1-1)) <br>

For example, in our project, the time scale is 2020 to 2099. In this chunk, the dates can be replaced.

```
tasmax = ds_max.tasmax.sel(time=slice("2020", "2099")).sel(**region).load()
```

<br>

---

**3. To clip dataset**<br> 
To clip the data according to region, ensure you have a shapefile available for the region. (See India Shapefiles folder for available files)