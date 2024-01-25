#!/bin/env python
import os
import sys

#sys.path.append('../python_pyhht/pyhht-dev/*')
#sys.path.append('../python_pyhht/pyhht-dev/pyhht')
sys.path.append('../python_download')
from MAIN_JSON import Get_var_name, Get_Date,distance_on_unit_sphere,ComputeVelocity,distance_on_unit_sphere_array
import pylab
from numpy import load,ones,where,array,std,mean,unique,sort,linspace,median,savez,pi,sin,cos,zeros,matmul,arange,diff,nanmean,nan
import scipy.io
import glob
#from ReadData_ICEX2020 import ReadData_ICEX2020,distance_on_unit_sphere,Get_time,ComputeVelocity
sys.path.append('../python_pyemd/PyEMD-master/*')
sys.path.append('../python_pyemd/PyEMD-master/')
from matplotlib.dates import date2num, num2date
from numpy import sqrt
from PyEMD import EMD,EEMD
#from pyemd import emd,eemd
## see https://pyemd.readthedocs.io/en/latest/examples.html#eemd
import itertools

### Basemap transformation
#from mpl_toolkits.basemap import Basemap

##### Triangulation and take the derivative
#####https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/trigradient_demo.html
from matplotlib.tri import (
    Triangulation, UniformTriRefiner, CubicTriInterpolator)
import time
from scipy.spatial import Delaunay

from numpy import arctan,cos,sin,sign,arctan2,isnan,argsort


def AreaOfTraingle(a,b,c):
	## Calculate area of triangle from three points
	# a, b, and c represent points of the triangle
	# a[0] = lon, a[1] = lat
	ab = distance_on_unit_sphere(a[1],a[0],b[1],b[0]) #distance_on_unit_sphere(lat[ft-1], lon[ft-1], lat[ft+1], lon[ft+1])
	bc = distance_on_unit_sphere(b[1],b[0],c[1],c[0])
	ca = distance_on_unit_sphere(c[1],c[0],a[1],a[0])

	### Use Heron's formula based on the Euclidian 
	s = (ab + bc + ca)/2.0
	Area = sqrt(s*(s-ab)*(s-bc)*(s-ca)) #m^2


	### http://mathforum.org/library/drmath/view/65316.html
	from numpy import tan,arctan,arccos
	R = 6373.0*1000.0 # raidus of the earth in m
	ab_norm = ab/R
	bc_norm = bc/R
	ca_norm = ca/R
	s_norm = (ab_norm + bc_norm + ca_norm)/2.0

	tanE4 = sqrt(tan(s_norm/2.0)*tan((s_norm-ab_norm)/2.0)*tan((s_norm-bc_norm)/2.0)*tan((s_norm-ca_norm)/2.0)) # tan(E/4) = sqrt(tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2))
	E = arctan(tanE4)*4.0
	AreaSphere = E*R*R # considering three points are on the sphere
	return Area, AreaSphere

def iStamper(ires):
	## Create my stam in the form 0000
	## if ires = 1 then 000001
	## if ires = 11 then 000011
	## if ires = 110 then 000110
	## if ires = 1124 the 001124
	st_ires = str(ires)
	nst = len(st_ires)
	if nst ==1:
		stamp = '00000'+st_ires
		
	elif nst ==2:
		stamp = '0000'+st_ires
			
	elif nst == 3:
		stamp = '000'+st_ires
			
	elif nst == 4:
		stamp = '00'+st_ires

	elif nst == 5:
		stamp = '0'+st_ires
			
	elif nst == 6:
		stamp = st_ires

	else:
		"ires needs to be less than 4 digits number"
		stamp = -1
		sys.exit(0)
					#print "Stamping is ", stamp
					
	return stamp

def NaNs(ntime):
	ar = zeros(ntime)
	ar[:] = nan
	return ar

import math 
  
# returns square of distance b/w two points  
def lengthSquare(X, Y):  
    xDiff = X[0] - Y[0]  
    yDiff = X[1] - Y[1]  
    return xDiff * xDiff + yDiff * yDiff 
      
def printAngle(A, B, C):  
	  
	# Square of lengths be a2, b2, c2  
	a2 = lengthSquare(B, C)  
	b2 = lengthSquare(A, C)  
	c2 = lengthSquare(A, B)  


		
	# length of sides be a, b, c  
	a = math.sqrt(a2);  
	b = math.sqrt(b2);  
	c = math.sqrt(c2);  

	s = (a+b+c)/2.0
	AreaHeron = sqrt(s*(s-a)*(s-b)*(s-c))

	# From Cosine law  
	alpha = math.acos((b2 + c2 - a2) /
	                     (2 * b * c));  
	betta = math.acos((a2 + c2 - b2) / 
	                     (2 * a * c));  
	gamma = math.acos((a2 + b2 - c2) / 
	                     (2 * a * b));  

	# Converting to degree  
	alpha = alpha * 180 / math.pi;  
	betta = betta * 180 / math.pi;  
	gamma = gamma * 180 / math.pi;  

	# printing all the angles  
	# print("alpha : %f" %(alpha))  
	# print("betta : %f" %(betta)) 
	# print("gamma : %f" %(gamma)) 

	return alpha,betta,gamma,AreaHeron


import cartopy.crs as ccrs
from numpy import meshgrid
def map_transform(lons,lats,lon_cent,lat_cent):
	x, y = meshgrid(lons, lats)
	use_proj= ccrs.LambertAzimuthalEqualArea(central_latitude=lat_cent, central_longitude=lon_cent);
	out_xyz = use_proj.transform_points(ccrs.Geodetic(), x, y)

	# separate x_array, y_array from the result(x,y,z) above
	x_array = out_xyz[:,:,0]
	y_array = out_xyz[:,:,1]

	xx = x_array[0,:]
	yy = y_array[:,0]

	return xx,yy


FOLDER = 'ICEX_2020'
#FOLDER = 'ICEX_2022'

FILE_Direc = '../python_download/main_SYNC_data/'+FOLDER+'/'

PLOT_FIGURE= 'NO'
#PLOT_FIGURE = 'YES'
if FOLDER=='ICEX_2020':
	fnameWB = FILE_Direc+ 'SYNC_uv_All_JAM-WB-0003.npz'
	fnameUT1 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0001.npz'
	fnameUT2 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0002.npz'
	fnameUT3 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0003.npz'
	fnameUT4 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0004.npz'
	fnameUT5 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0005.npz'

	fnameIT1 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0001.npz'
	fnameIT2 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0002.npz'
	fnameIT3 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0003.npz'
	fnameIT4 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0004.npz'


	W = dict(load(fnameWB))
	U1 = dict(load(fnameUT1))
	U2 = dict(load(fnameUT2))
	U3 = dict(load(fnameUT3))
	U4 = dict(load(fnameUT4))
	U5 = dict(load(fnameUT5))

	I1 = dict(load(fnameIT1))
	I2 = dict(load(fnameIT2))
	I3 = dict(load(fnameIT3))
	I4 = dict(load(fnameIT4))

	W['DeviceCode'] = 'W'
	U1['DeviceCode'] = 'U1' 
	U2['DeviceCode'] = 'U2' 
	U3['DeviceCode'] = 'U3' 
	U4['DeviceCode'] = 'U4' 
	U5['DeviceCode'] = 'U5' 

	I1['DeviceCode'] = 'I1' 
	I2['DeviceCode'] = 'I2' 
	I3['DeviceCode'] = 'I3' 
	I4['DeviceCode'] = 'I4' 


	import itertools
	dictionary_ar = [
	'W',
	'U1',
	'U2',
	'U3',
	'U4',
	'U5',
	'I1','I2','I3','I4',
	]  


elif FOLDER=='ICEX_2022':
	fnameWB = FILE_Direc+ 'SYNC_uv_All_JAM-WB-0004.npz'
	fnameUT6 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0006.npz'
	fnameUT7 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0007.npz'
	fnameUT8 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0008.npz'
	fnameUT9 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0009.npz'
	fnameUT10 =FILE_Direc+ 'SYNC_uv_All_JAM-UT-0010.npz'

	fnameIT6 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0006.npz'
	fnameIT7 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0007.npz'
	fnameIT8 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0008.npz'
	fnameIT9 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0009.npz'
	fnameIT10 =FILE_Direc+ 'SYNC_uv_All_JAM-IT-0010.npz'

	W4 = dict(load(fnameWB))
	U6 = dict(load(fnameUT6))
	U7 = dict(load(fnameUT7))
	U8 = dict(load(fnameUT8))
	U9 = dict(load(fnameUT9))
	U10 = dict(load(fnameUT10))

	I6 = dict(load(fnameIT6))
	I7 = dict(load(fnameIT7))
	I8 = dict(load(fnameIT8))
	I9 = dict(load(fnameIT9))
	I10 = dict(load(fnameIT10))

	W4['DeviceCode'] = 'W4'
	U6['DeviceCode'] = 'U6' 
	U7['DeviceCode'] = 'U7' 
	U8['DeviceCode'] = 'U8' 
	U9['DeviceCode'] = 'U9' 
	U10['DeviceCode'] = 'U10' 

	I6['DeviceCode'] = 'I6' 
	I7['DeviceCode'] = 'I7' 
	I8['DeviceCode'] = 'I8' 
	I9['DeviceCode'] = 'I9' 
	I10['DeviceCode'] = 'I10' 

	W4['FILENAME'] = fnameWB
	U6['FILENAME'] = fnameUT6
	U7['FILENAME'] = fnameUT7
	U8['FILENAME'] = fnameUT8
	U9['FILENAME'] = fnameUT9 
	U10['FILENAME'] = fnameUT10

	I6['FILENAME'] = fnameIT6
	I7['FILENAME'] = fnameIT7 
	I8['FILENAME'] = fnameIT8 
	I9['FILENAME'] = fnameIT9
	I10['FILENAME'] = fnameIT10

	dictionary_ar = [
	'W4',
	'U6',
	'U7',
	'U8',
	'U9',
	'U10',
	'I6','I7','I8','I9','I10'
	]  


tri_combo = list(itertools.combinations(dictionary_ar, 3))
ntriangles_combo = len(tri_combo)
# program = 'A='+tri_combo[ftri][0]
# program = 'B='+tri_combo[ftri][1]
# program = 'C='+tri_combo[ftri][2]
# print(program)
# exec(program)
print('Total triangles',ntriangles_combo)

#time_interval_ind_ar = (1,3,6,12,24,48)
#time_interval_ind_ar = (72,96,120,144)
#time_interval_ind_ar = (1,3,6,12,24,48)
time_interval_ind_ar = (6,12,24,48,72,96,120,144)
#time_interval_ind_ar = (48,72,96,120,144)
ninterval = len(time_interval_ind_ar)
#print(ntriangles_combo)
#sys.exit()
#### Pick three points in a triangle, A, B, and C
#time_interval_ind = 1 #tau_step=1: (F[i+time_interval_ind] - F[i-time_interval_ind])/(t[i+time_interval_ind] - t[i-time_interval_ind])
#time_interval_ind = 12 # 6hourly
#time_interval_ind = 24 # 12hourly
# time_interval_ind = 48 # 24hourly

for my_interval in range(0,ninterval,1):
	time_interval_ind = time_interval_ind_ar[my_interval]

	#savedirec = './main_Deformation_trig_all/'+FOLDER+'/TimeIntervalStep_'+str(time_interval_ind)

	#savedirec = '/Volumes/Corvegas/ICEX2022_DATA/python_deformation_itkin/main_Deformation_trig_all/'+FOLDER+'/TimeIntervalStep_'+str(time_interval_ind)
	savedirec = '/Users/toshi/Projects/ICEX_tmp/python_deformation_itkin/main_Deformation_trig_all/'+FOLDER+'/TimeIntervalStep_'+str(time_interval_ind)
	try:
		os.stat(savedirec)
	except:
		os.mkdir(savedirec)



	for ftri in range(0,ntriangles_combo,1):
		print('Computing the triangle ',tri_combo[ftri],ftri)

		programA = 'A='+tri_combo[ftri][0]
		programB = 'B='+tri_combo[ftri][1]
		programC = 'C='+tri_combo[ftri][2]

		exec(programA)
		exec(programB)
		exec(programC)

		DeviceName_a = A['DeviceName']
		DeviceName_b = B['DeviceName']
		DeviceName_c = C['DeviceName']

		#DeviceCode_ar_sort = sort((A['DeviceCode'],B['DeviceCode'],C['DeviceCode']))
		######### 
		# tstmp = iStamper(time_interval_ind)
		# savefname = './main_ThreePoints_All/Data_'+DeviceCode_ar[0]+'_'+DeviceCode_ar[1]+'_'+DeviceCode_ar[2]+'_Tinterval_'+tstmp
		
		#pylab.close()

		tt_sync = A['tt_sync']
		tt_sec_sync = A['tt_sec_sync']

		# urx = 5850000# 3750000#+x0
		# ury = 5850000#+y0

		# lat_ts = nanmean(A['lat_sync']+B['lat_sync']+C['lat_sync'])/3.0
		# map_transform = Basemap(
		#             width=urx,height=ury,
		#             resolution='l',projection='stere',\
		#             lat_ts=lat_ts,lat_0=90,lon_0=-45.0,)




		ntime = len(tt_sync)


		
		# ## Declare the variables
		# u_a =  NaNs(ntime)
		# v_a =  NaNs(ntime)
		# speed_a =  NaNs(ntime)

		# u_b =  NaNs(ntime)
		# v_b =  NaNs(ntime)
		# speed_b =  NaNs(ntime)

		# u_c =  NaNs(ntime)
		# v_c =  NaNs(ntime)
		# speed_c =  NaNs(ntime)

		# Area =  NaNs(ntime)
		# AreaSphere =  NaNs(ntime)

		lon_centroid = (A['lon_sync'] + B['lon_sync'] + C['lon_sync'])/3.0
		lat_centroid = (A['lat_sync'] + B['lat_sync'] + C['lat_sync'])/3.0

		# u_centroid =  NaNs(ntime)
		# v_centroid =  NaNs(ntime)
		# ux_centroid =  NaNs(ntime)
		# vx_centroid =  NaNs(ntime)
		# uy_centroid =  NaNs(ntime)
		# vy_centroid =  NaNs(ntime)


		start_time = time.time()
		for ft in range(time_interval_ind,ntime-time_interval_ind,1):
		#for ft in range(6225,ntime-time_interval_ind,1):
			print('Percent, 1st loop, 2nd loop, 3rd loop = ',100*my_interval/ninterval,100*ftri/ntriangles_combo,100*ft/ntime)
			
			print('Start date = ',num2date(tt_sync[ft]))
			#sys.exit()
			#distance_on_unit_sphere(lat1, long1, lat2, long2)
			T0 = tt_sec_sync[ft-time_interval_ind]
			T1 = tt_sec_sync[ft+time_interval_ind]
			# lon0 = A['lon_sync'][ft-dt]
			# lat0 = A['lat_sync'][ft-dt]
			# lon1 = A['lon_sync'][ft+dt]
			# lat1 = A['lat_sync'][ft+dt]
			# #u_a[ft],v_a[ft],speed_a[ft] = ComputeVelocity(T0,T1,lon0,lat0,lon1,lat1)

			LON_LAT_ALL = (A['lon_sync'][ft-time_interval_ind],A['lat_sync'][ft-time_interval_ind],A['lon_sync'][ft+time_interval_ind],A['lat_sync'][ft+time_interval_ind],
			B['lon_sync'][ft-time_interval_ind],B['lat_sync'][ft-time_interval_ind],B['lon_sync'][ft+time_interval_ind],B['lat_sync'][ft+time_interval_ind],
			C['lon_sync'][ft-time_interval_ind],C['lat_sync'][ft-time_interval_ind],C['lon_sync'][ft+time_interval_ind],C['lat_sync'][ft+time_interval_ind],
			A['lon_sync'][ft],B['lon_sync'][ft],C['lon_sync'][ft],
			A['lat_sync'][ft],B['lat_sync'][ft],C['lat_sync'][ft],
			)

			
			if isnan(LON_LAT_ALL).any()==True:
				print('there is a nan, so I skip')
			else:

				##### Compute velocity at time = ft
				u_a,v_a,speed_a = ComputeVelocity(T0,T1,A['lon_sync'][ft-time_interval_ind],A['lat_sync'][ft-time_interval_ind],A['lon_sync'][ft+time_interval_ind],A['lat_sync'][ft+time_interval_ind])
				u_b,v_b,speed_b = ComputeVelocity(T0,T1,B['lon_sync'][ft-time_interval_ind],B['lat_sync'][ft-time_interval_ind],B['lon_sync'][ft+time_interval_ind],B['lat_sync'][ft+time_interval_ind])
				u_c,v_c,speed_c = ComputeVelocity(T0,T1,C['lon_sync'][ft-time_interval_ind],C['lat_sync'][ft-time_interval_ind],C['lon_sync'][ft+time_interval_ind],C['lat_sync'][ft+time_interval_ind])


				u_ar = array((u_a,u_b,u_c))
				v_ar = array((v_a,v_b,v_c))

				lon_ar = array((A['lon_sync'][ft],B['lon_sync'][ft],C['lon_sync'][ft],))
				lat_ar = array((A['lat_sync'][ft],B['lat_sync'][ft],C['lat_sync'][ft],))

				#lon_cent = 
				xx_ar,yy_ar = map_transform(lon_ar,lat_ar,lon_centroid[ft],lat_centroid[ft])

				#xx_ar, yy_ar = map_transform(lon_ar,lat_ar)
				#xx_centroid,yy_centroid = map_transform(lon_centroid[ft],lat_centroid[ft]) 
				xx_centroid,yy_centroid = map_transform(lon_centroid[ft],lat_centroid[ft],lon_centroid[ft],lat_centroid[ft])

				#sys.exit()
				DeviceCode_ar = ((A['DeviceCode'],B['DeviceCode'],C['DeviceCode']))

				pt1 = array([\
					[xx_ar[0],yy_ar[0]],
					[xx_ar[1],yy_ar[1]],
					[xx_ar[2],yy_ar[2]],
					])

				pt1_LonLat = array([\
					[lon_ar[0],lat_ar[0]],
					[lon_ar[1],lat_ar[1]],
					[lon_ar[2],lat_ar[2]],
					])

				Area, AreaSphere = AreaOfTraingle(pt1_LonLat[0],pt1_LonLat[1],pt1_LonLat[2])

				### reorder the points in counter clockwise
				theta_centroid = arctan2(yy_ar - yy_centroid, xx_ar- xx_centroid) * 180 / pi
				ind_counter = argsort(theta_centroid)

				pt1 = pt1[ind_counter]
				pt1_LonLat = pt1_LonLat[ind_counter]
				DeviceCode_ar = (DeviceCode_ar[ind_counter[0]],DeviceCode_ar[ind_counter[1]],DeviceCode_ar[ind_counter[2]])
				DeviceCode_stamp = sort(DeviceCode_ar)
				u_ar = u_ar[ind_counter]
				v_ar = v_ar[ind_counter]
				xx_ar = xx_ar[ind_counter]
				yy_ar = yy_ar[ind_counter]
				lon_ar = lon_ar[ind_counter]
				lat_ar = lat_ar[ind_counter]

				alpha,beta,gamma,AreaHeron = printAngle(pt1[0], pt1[1], pt1[2])
				Angles = array((alpha,beta,gamma))
				a = Area
				u1 = u_ar[0]
				u2 = u_ar[1]
				u3 = u_ar[2]
				v1 = v_ar[0]
				v2 = v_ar[1]
				v3 = v_ar[2]	
				x1 = xx_ar[0]
				x2 = xx_ar[1]
				x3 = xx_ar[2]
				y1 = yy_ar[0]
				y2 = yy_ar[1]
				y3 = yy_ar[2]	

				ux = (0.5/a)*(\
					(u1+u3)*(y1-y3)\
					+(u1+u2)*(y1-y2)\
					+(u2+u3)*(y2-y3)\
					)

				uy = (-0.5/a)*(\
				(u1+u3)*(x1-x3)\
				+(u1+u2)*(x1-x2)\
				+(u2+u3)*(x2-x3)\
				)

				vx = (0.5/a)*(\
					(v1+v3)*(y1-y3)\
					+(v1+v2)*(y1-y2)\
					+(v2+v3)*(y2-y3)\
					)

				vy = (-0.5/a)*(\
				(v1+v3)*(x1-x3)\
				+(v1+v2)*(x1-x2)\
				+(v2+v3)*(x2-x3)\
				)

				e11 = ux
				e22 = vy
				e12 = 0.5*(uy+vx)
				e21 = 0.5*(vx+uy)

				div = e11 + e22
				shr= sqrt((e11-e22)**2 + (e12+e21)**2) # sqrt( (ux-vy)**2 + (uy + vx)**2)
				tot = sqrt(div**2+shr**2)
				
				
				F = dict()
				F['ftri'] = ftri
				F['time_interval_ind'] = time_interval_ind
				F['tt'] = A['tt_sync'][ft]
				F['tt_sec'] =  A['tt_sec_sync'][ft]
				#F['map_transform'] = map_transform
				F['Area'] = Area
				F['AreaSphere'] = AreaSphere
				F['lon_centroid'] = lon_centroid[ft]
				F['lat_centorid'] = lat_centroid[ft]
				F['Angles'] = Angles
				F['lon_a'] = A['lon_sync'][ft]
				F['lat_a'] = A['lat_sync'][ft]

				F['lon_b'] = B['lon_sync'][ft]
				F['lat_b'] = B['lat_sync'][ft]
				
				F['lon_c'] = C['lon_sync'][ft]
				F['lat_c'] = C['lat_sync'][ft]



				F['DeviceName_a'] = DeviceName_a
				F['u_a'] = u_a
				F['v_a'] = v_a
				F['speed_a'] = speed_a

				F['DeviceName_b'] = DeviceName_b
				F['u_b'] = u_b
				F['v_b'] = v_b
				F['speed_b'] = speed_b

				F['DeviceName_c'] = DeviceName_c
				F['u_c'] = u_c
				F['v_c'] = v_c
				F['speed_c'] = speed_c

				### Centroid
				
				F['ux'] = ux
				F['vx'] = vx
				F['uy'] = uy
				F['vy'] = vy

				F['e11'] = e11
				F['e22'] = e22
				F['e12'] = e12
				F['e21'] = e21

				F['div'] = div
				F['shr'] = shr
				F['tot'] = tot
				### create the directory for each time step
				tstmp = iStamper(ft)
				savedirec_t = savedirec+'/Timestep_'+tstmp
				try:
					os.stat(savedirec_t)
				except:
					os.mkdir(savedirec_t)

				
				savefname = savedirec_t+'/Data_'+DeviceCode_stamp[0]+'_'+DeviceCode_stamp[1]+'_'+DeviceCode_stamp[2]
				savez(savefname,**F)

				if PLOT_FIGURE=='YES':
					savefnameFIG = savedirec_t+'/Trig_'+DeviceCode_stamp[0]+'_'+DeviceCode_stamp[1]+'_'+DeviceCode_stamp[2]
					### First plot all the initial GPS coordinate
					###########################
					fsize=18
					figW = 8*1.2 #8*1.5
					figH = 11*1 #11*1.5 #11
					legend_properties = {'weight':'bold'}

					fig1 = pylab.figure(0,figsize=(figW,figH))
					params = {'backend': 'ps',
					                            'axes.labelsize': fsize,
					#                            'text.fontsize': fsize,
					                            'legend.fontsize': fsize,
					                            'xtick.labelsize': fsize,
					#                            'title.fontsize': fsize,
					                            'ytick.labelsize': fsize,
					                            'font.weight': 'bold'}
					pylab.rcParams.update(params)
					msize = 23
					ax = pylab.subplot(1,1,1)
					ax.plot(W['lon_sync'][ft],W['lat_sync'][ft],'m*',markersize=msize,label='Warm buoy')

					## UTs
					ax.plot(U1['lon_sync'][ft],U1['lat_sync'][ft],'ro',markersize=msize,label='U1')
					ax.plot(U2['lon_sync'][ft],U2['lat_sync'][ft],'bo',markersize=msize,label='U2')
					ax.plot(U3['lon_sync'][ft],U3['lat_sync'][ft],'go',markersize=msize,label='U3')
					ax.plot(U4['lon_sync'][ft],U4['lat_sync'][ft],'co',markersize=msize,label='U4')
					ax.plot(U5['lon_sync'][ft],U5['lat_sync'][ft],'yo',markersize=msize,label='U5')

					## ITs
					ax.plot(I1['lon_sync'][ft],I1['lat_sync'][ft],'rd',markersize=msize,label='I1')
					ax.plot(I2['lon_sync'][ft],I2['lat_sync'][ft],'bd',markersize=msize,label='I2')
					ax.plot(I3['lon_sync'][ft],I3['lat_sync'][ft],'gd',markersize=msize,label='I3')
					ax.plot(I4['lon_sync'][ft],I4['lat_sync'][ft],'cd',markersize=msize,label='I4')

					ax.plot(A['lon_sync'][ft],A['lat_sync'][ft],'ko',label='A')
					ax.plot(B['lon_sync'][ft],B['lat_sync'][ft],'k*',label='B')
					ax.plot(C['lon_sync'][ft],C['lat_sync'][ft],'k+',label='C')

					leg = ax.legend(loc='upper left', shadow=True,fontsize=fsize,ncol=2,fancybox=True, borderaxespad=0.)


					from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
					from mpl_toolkits.axes_grid1.inset_locator import mark_inset

					ax2 = zoomed_inset_axes(ax, 3, loc=1) # zoom = 6
					ax2.plot(W['lon_sync'][ft],W['lat_sync'][ft],'m*',markersize=msize,label='Warm buoy')

					## UTs
					ax2.plot(U1['lon_sync'][ft],U1['lat_sync'][ft],'ro',markersize=msize,label='U1')
					ax2.plot(U2['lon_sync'][ft],U2['lat_sync'][ft],'bo',markersize=msize,label='U2')
					ax2.plot(U3['lon_sync'][ft],U3['lat_sync'][ft],'go',markersize=msize,label='U3')
					ax2.plot(U4['lon_sync'][ft],U4['lat_sync'][ft],'co',markersize=msize,label='U4')
					ax2.plot(U5['lon_sync'][ft],U5['lat_sync'][ft],'yo',markersize=msize,label='U5')


					ax2.plot(A['lon_sync'][ft],A['lat_sync'][ft],'ko',label='A')
					ax2.plot(B['lon_sync'][ft],B['lat_sync'][ft],'k*',label='B')
					ax2.plot(C['lon_sync'][ft],C['lat_sync'][ft],'k+',label='C')

					lon_UTs = array((U1['lon_sync'][ft],U2['lon_sync'][ft],U3['lon_sync'][ft],U4['lon_sync'][ft],U5['lon_sync'][ft],))
					lat_UTs = array((U1['lat_sync'][ft],U2['lat_sync'][ft],U3['lat_sync'][ft],U4['lat_sync'][ft],U5['lat_sync'][ft],))

					#ax2.set_xlim((-140.89,-140.85))
					#ax2.set_ylim((71.125,71.135))
					ax2.set_xlim(min(lon_UTs)-0.01,max(lon_UTs)+0.01)
					ax2.set_ylim(min(lat_UTs)-0.005,max(lat_UTs)+0.005)

					ax2.set_xticks([])
					ax2.set_yticks([])

					# draw a bbox of the region of the inset axes in the parent axes and
					# connecting lines between the bbox and the inset axes area
					mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec="0.5")
					fig1.savefig(savefnameFIG,dpi=50)

					#pylab.show()
					#sys.exit()

					pylab.close()
			#pylab.show()


			
			#sys.exit(0)

		## end the time loop
		end_time = time.time()
		tot_time = end_time - start_time
		print('Time to complete the time loop: ',tot_time)

		

# pylab.figure
# ax = pylab.subplot(4,1,1)
# ax.plot(A['tt_sync'],A['u_sync'],'k-')
# ax.plot(tt_sync,u_a,'r-')
# ax.plot(A['tt'],A['u'],'b-')

# ax = pylab.subplot(4,1,2)
# ax.plot(A['tt_sync'],A['speed_sync'],'k-')
# ax.plot(tt_sync,speed_a,'r-')
# ax.plot(A['tt'],A['speed'],'b-')
# ax.plot(A['tt'],A['speed_norm'],'g-')


# ax = pylab.subplot(4,1,3)
# ax.plot(A['tt'],A['lon'],'b-')
# ax.plot(A['tt_sync'],A['lon_sync'],'r-')

# ax = pylab.subplot(4,1,4)
# ax.plot(A['tt'],A['lat'],'b-')
# ax.plot(A['tt_sync'],A['lat_sync'],'r-')




# pylab.show()















