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
from os.path import exists

FOLDER = 'ICEX_2020'
FOLDER = 'ICEX_2022'


TimeStepFolder = 'TimeIntervalStep_1'
TimeStepFolder = 'TimeIntervalStep_3'

FOLDER_ar = [ 'ICEX_2020', 'ICEX_2022']
TimeStepFolder_ar = ['TimeIntervalStep_3','TimeIntervalStep_6','TimeIntervalStep_12','TimeIntervalStep_24','TimeIntervalStep_48','TimeIntervalStep_72','TimeIntervalStep_96','TimeIntervalStep_120','TimeIntervalStep_144']

nFolders = len(FOLDER_ar)
nSteps = len(TimeStepFolder_ar)

for fold in range(0,nFolders,1):
	FOLDER = FOLDER_ar[fold]
	for fstep in range(0,nSteps,1):
		TimeStepFolder=TimeStepFolder_ar[fstep]

		tFOLDER = './main_Deformation_trig_all/'+FOLDER+'/'+TimeStepFolder+'/Timestep_*'
		#savedirec = '/Users/toshi/Projects/ICEX_tmp/python_deformation_itkin/main_Deformation_trig_all/'+FOLDER+'/TimeIntervalStep_'+str(time_interval_ind)
		tFOLDER = '/Users/toshi/Projects/ICEX_tmp/python_deformation_itkin/main_Deformation_trig_all/'+FOLDER+'/'+TimeStepFolder+'/Timestep_*'
		import itertools
		if FOLDER=='ICEX_2020':
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
			dictionary_ar = [
			'W4',
			'U6',
			'U7',
			'U8',
			'U9',
			'U10',
			'I6','I7','I8','I9','I10'
			]


		savedirec = './main_trig_timeseries_all/'+FOLDER+'/'+TimeStepFolder
		try:
			os.stat(savedirec)
		except:
			os.mkdir(savedirec)
			
		tri_combo = list(itertools.combinations(dictionary_ar, 3))
		ntriangles_combo = len(tri_combo)

		FolderList =sort(glob.glob(tFOLDER))
		nF = len(FolderList)
		#sys.exit()
		start_time = time.time()
		### declare filename

		#sys.exit()
		for ftr in range(0,ntriangles_combo,1):

			tri_name = sort(tri_combo[ftr])
			FileType = 'Data_'+tri_name[0]+'_'+tri_name[1]+'_'+tri_name[2]+'.npz'

			### Does this file exist in the first folder that we will search
			filename = FolderList[0]+'/'+FileType
			file_exists = exists(filename)

			#if file_exists==False:
			#	print('No file in the first folder, the naming method is not correct?')
			#	sys.exit()



			T = dict() ## time series
			T['div'] = []
			T['shr'] = []
			T['tot'] = []
			T['tt'] = []
			T['Area'] = []
			T['MininumAngle_Trig'] = []
			T['vort'] = []

			T['lon_a'] = []
			T['lat_a'] = []

			T['lon_b'] = []
			T['lat_b'] = []

			T['lon_c'] = []
			T['lat_c'] = []

			T['lon'] = []
			T['lat'] = []

			T['FileType'] = []
			T['ftri'] = []

			T['ux'] = []
			T['uy'] = []
			T['vx'] = []
			T['vy'] = []

			fid = 1

			for fid in range(0,nF,1):
				print(fid/nF,FOLDER,fold/nFolders,TimeStepFolder,fstep/nSteps)
				direc = FolderList[fid]

				filename = direc+'/'+FileType

				file_exists = exists(filename)

				if file_exists == True:
					F = load(filename)

					T['tt'].append(float(F['tt']))
					T['shr'].append(float(F['shr']))
					T['div'].append(float(F['div']))
					T['tot'].append(float(F['tot']))
					T['Area'].append(float(F['Area']))

					T['MininumAngle_Trig'].append(min(F['Angles']))

					xi = float(F['vx']) - float(F['uy'])
					T['vort'].append(xi)

					T['lon_a'].append(float(F['lon_a']))
					T['lat_a'].append(float(F['lat_a']))

					T['lon_b'].append(float(F['lon_b']))
					T['lat_b'].append(float(F['lat_b']))


					T['lon_c'].append(float(F['lon_c']))
					T['lat_c'].append(float(F['lat_c']))
					T['FileType'].append(FileType)

					T['ftri'].append(float(F['ftri']))

					T['lon'].append(float(F['lon_centroid']))
					T['lat'].append(float(F['lat_centorid']))

					T['ux'].append(float(F['ux']))
					T['uy'].append(float(F['uy']))
					T['vx'].append(float(F['vx']))
					T['vy'].append(float(F['vy']))


		#			if fid==3:
		#				sys.exit()
				else:
					print('No more data, so I terminate the loop')
					#break

			FTRI = int(unique(T['ftri']))
			T['DeviceName_a'] = F['DeviceName_a']
			T['DeviceName_b'] = F['DeviceName_b']
			T['DeviceName_c'] = F['DeviceName_c']
			#sys.exit()
			savefname = savedirec+'/'+FileType.split('.')[0]+'_ftri_'+str(FTRI)+'.npz'

			savez(savefname,**T)
			#sys.exit()
		end_time = time.time()
			#sys.exit()
