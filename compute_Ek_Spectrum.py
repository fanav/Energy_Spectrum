# python 3

# Script for the computation of 3D spectrum of the Total Kinetic Energy
# Adapted to the Taylor-Green vortex (TGV) problem.
# CREATED by FARSHAD NAVAH
# McGill University
# farshad.navah .a.t. mail.mcgill.ca
# 2018
# provided as is with no garantee.
# Please cite:
#    https://github.com/fanav/Energy_Spectrum
#    https://arxiv.org/abs/1809.03966

# -----------------------------------------------------------------
#  IMPORTS - ENVIRONMENT
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt

# -----------------------------------------------------------------
#  TGV QUANTS
# -----------------------------------------------------------------

# These quantities are to account for particular non-dimensionalizations of state variables.
# In general U0=1.
c  = sqrt(1.4);
Ma = 0.1;
U0 = Ma*c; 

# -----------------------------------------------------------------
#  INPUT FILE PARAMS
# -----------------------------------------------------------------

data_path = "./"
file = input("Desired file name? ") # enter velocityfld_ascii.dat for demo.

# -----------------------------------------------------------------
#  OUTOUT FILE PARAMS
# -----------------------------------------------------------------

Figs_Path = "./"
Fig_file_name = "Ek_Spectrum"

# -----------------------------------------------------------------
#  READ FILES
# -----------------------------------------------------------------

localtime = time.asctime( time.localtime(time.time()) )
print ("\nReading files...localtime",localtime)

#load the ascii file
data     = np.loadtxt(data_path+file, skiprows=2)

print ("shape of data = ",data.shape)

localtime = time.asctime( time.localtime(time.time()) )
print ("Reading files...localtime",localtime, "- END\n")

# -----------------------------------------------------------------
#  COMPUTATIONS
# -----------------------------------------------------------------
localtime = time.asctime( time.localtime(time.time()) )
print ("Computing spectrum... ",localtime)

N = int(round((len(data)**(1./3))))
print("N =",N)
eps = 1e-50 # to void log(0)

U = data[:,3].reshape(N,N,N)/U0
V = data[:,4].reshape(N,N,N)/U0
W = data[:,5].reshape(N,N,N)/U0

amplsU = abs(np.fft.fftn(U)/U.size)
amplsV = abs(np.fft.fftn(V)/V.size)
amplsW = abs(np.fft.fftn(W)/W.size)

EK_U  = amplsU**2
EK_V  = amplsV**2 
EK_W  = amplsW**2 

EK_U = np.fft.fftshift(EK_U)
EK_V = np.fft.fftshift(EK_V)
EK_W = np.fft.fftshift(EK_W)

sign_sizex = np.shape(EK_U)[0]
sign_sizey = np.shape(EK_U)[1]
sign_sizez = np.shape(EK_U)[2]

box_sidex = sign_sizex
box_sidey = sign_sizey
box_sidez = sign_sizez

box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2+(box_sidez)**2))/2.)+1)

centerx = int(box_sidex/2)
centery = int(box_sidey/2)
centerz = int(box_sidez/2)

print ("box sidex     =",box_sidex) 
print ("box sidey     =",box_sidey) 
print ("box sidez     =",box_sidez)
print ("sphere radius =",box_radius )
print ("centerbox     =",centerx)
print ("centerboy     =",centery)
print ("centerboz     =",centerz,"\n" )
	            
EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius
EK_W_avsphr = np.zeros(box_radius,)+eps ## size of the radius

for i in range(box_sidex):
	for j in range(box_sidey):
		for k in range(box_sidez):            
			wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2+(k-centerz)**2)))
			EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j,k]
			EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j,k]    
			EK_W_avsphr[wn] = EK_W_avsphr [wn] + EK_W [i,j,k]        

EK_avsphr = 0.5*(EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)
	                      
fig = plt.figure()
plt.title("Kinetic Energy Spectrum")
plt.xlabel(r"k (wavenumber)")
plt.ylabel(r"TKE of the k$^{th}$ wavenumber")

realsize = len(np.fft.rfft(U[:,0,0]))
plt.loglog(np.arange(0,realsize),((EK_avsphr[0:realsize] )),'k')
plt.loglog(np.arange(realsize,len(EK_avsphr),1),((EK_avsphr[realsize:] )),'k--')
axes = plt.gca()
axes.set_ylim([10**-25,5**-1])

print("Real      Kmax    = ",realsize)
print("Spherical Kmax    = ",len(EK_avsphr))

TKEofmean_discrete = 0.5*(np.sum(U/U.size)**2+np.sum(V/V.size)**2+np.sum(W/W.size)**2)
TKEofmean_sphere   = EK_avsphr[0]

total_TKE_discrete = np.sum(0.5*(U**2+V**2+W**2))/(N*1.0)**3
total_TKE_sphere   = np.sum(EK_avsphr)

print("the KE  of the mean velocity discrete  = ",TKEofmean_discrete)
print("the KE  of the mean velocity sphere    = ",TKEofmean_sphere )
print("the mean KE discrete  = ",total_TKE_discrete)
print("the mean KE sphere    = ",total_TKE_sphere)

localtime = time.asctime( time.localtime(time.time()) )
print ("Computing spectrum... ",localtime, "- END \n")

# -----------------------------------------------------------------
#  OUTPUT/PLOTS
# -----------------------------------------------------------------

dataout      = np.zeros((box_radius,2)) 
dataout[:,0] = np.arange(0,len(dataout))
dataout[:,1] = EK_avsphr[0:len(dataout)]

np.savetxt(Figs_Path+Fig_file_name+'.dat',dataout)
fig.savefig(Figs_Path+Fig_file_name+'.pdf')
plt.show()


