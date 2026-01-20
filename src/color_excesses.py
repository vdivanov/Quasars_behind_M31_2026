###############################################################
# Code to calculate the color excess and the absorption for
# quasars behind M31, with respect to a reference sample of
# field quasars outside M31 (Jan 2026).
###############################################################

from astropy.io import fits, ascii
from astropy.stats import median_absolute_deviation as mad
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

###############################################################
# function to calculate an weighted average and its error
#
def weighted_avg_and_std(values, weights):
    # Return the weighted average and standard deviation.
    # values, weights -- NumPy ndarrays with the same shape.
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

###############################################################
# Parameters:

# Input fits tables:
# -- quasars behind M31
NameM31Quasars = '../data/quasars_M31.fits'
# -- reference field Quaia quasars:
NameRefQuasars = '../data/quasars_ref.fits'

# Output files:
NameM31QuasarsOut = 'quasars_M31_out.fits'
# Other output ascii files color_xx0.dat contain the derived
# intrinsic colors for the respective filters, the rms and the
# number of quasars in each redshift bin.

# output plot names; pdfs are also generated:
Name_plot1 = 'colors_dered.png'
Name_plot2 = 'Av_quasars_vs_Av_maps.png'

Err_max = 0.1  # upper limit on photometric errors for reference
# quasars [mag]

RA_m31, DEC_m31  = 10.6847, 41.2688  # 2MASS M31 center [degr]

Ebv_M31_0, Ebv_err_M31_0 = 0.060, 0.010 # foreground Milky Way
# extinction forwards M31 and its error [mag]

Flag_MW_Av = 1  # to correct for the foreground MW extinction:
# 1 - yes and 0 - no.

Rv = 3.1  # ratio of relative to total-to-selective extinction ratio

dZ_max = 0.1  # Half-width of the window along redshit within which
# to select quasars from the reference sample for the determination
# of the intrinsic color of each quasar behind M31 [dex].

Bins_rho = np.array([1.5, 3.265, 5.05, 7.08, 11.0]) # radial
# distance intervals; for plotting [arcmin].

# Reddening law:
AgAv = 1.1792
ArAv = 0.8781
AiAv = 0.6686
AzAv = 0.5118
AyAv = 0.4295

###############################################################

# Reading the sample of reference quasars away from M31.
hdul = fits.open(NameRefQuasars)
hdul.info()
cols = hdul[1].columns
cols.info()
data = hdul[1].data

RA         = data['RA'].T  # coordinates RA and DEC
DEC        = data['DEC'].T
Rho_D25    = data['rho/R25'].T  # distance om M31 centre in units of R25
z          = data['z'].T   # Quaia redshift and error
z_err      = data['z_err'].T
g_mag      = data['g_mag'].T   # Pan-STARSS apparent magnitudes and errors
g_mag_err  = data['g_mag_err'].T
r_mag      = data['r_mag'].T
r_mag_err  = data['r_mag_err'].T
i_mag      = data['i_mag'].T
i_mag_err  = data['i_mag_err'].T
z_mag      = data['z_mag'].T
z_mag_err  = data['z_mag_err'].T
y_mag      = data['y_mag'].T
y_mag_err  = data['y_mag_err'].T
Ebv        = data['E(B-V)'].T  # Milky Way extinction from Schlafly et al.
print('Number of reference quasars = ', len(RA))

Ag = AgAv * Rv * Ebv
Ar = ArAv * Rv * Ebv
Ai = AiAv * Rv * Ebv
Az = AzAv * Rv * Ebv
Ay = AyAv * Rv * Ebv
Egr =  Ag - Ar
Egi =  Ag - Ai
Egz =  Ag - Az
Egy =  Ag - Ay
g_mag_c    = g_mag - Ag
r_mag_c    = r_mag - Ar
i_mag_c    = i_mag - Ai
z_mag_c    = z_mag - Az
y_mag_c    = y_mag - Ay

gr_mag     = g_mag - r_mag
gr_mag_c   = g_mag - r_mag - Egr
gr_mag_err = np.sqrt(g_mag_err**2 + r_mag_err**2)

gi_mag     = g_mag - i_mag
gi_mag_c   = g_mag - i_mag - Egi
gi_mag_err = np.sqrt(g_mag_err**2 + i_mag_err**2)

gz_mag     = g_mag - z_mag
gz_mag_c   = g_mag - z_mag - Egz
gz_mag_err = np.sqrt(g_mag_err**2 + z_mag_err**2)

gy_mag     = g_mag - y_mag
gy_mag_c   = g_mag - y_mag - Egy
gy_mag_err = np.sqrt(g_mag_err**2 + y_mag_err**2)


###############################################################

# Reading the sample of quasars behind M31.
hdul = fits.open(NameM31Quasars)
hdul.info()
cols = hdul[1].columns
cols.info()
data = hdul[1].data

PS1_ID_M31       = np.array(data['PS1_ID'].T, dtype=object)  # Pan-STARSS ID ????
Gaia_DR3_ID      = np.array(data['Gaia_DR3_ID'].T, dtype=object)  # Gaia ID
RA_M31           = np.array(data['RA'].T, dtype=np.float64)  # coordinates RA and DEC
DEC_M31          = np.array(data['DEC'].T, dtype=np.float64)
Rho_D25_M31      = np.array(data['rho/R25'].T, dtype=np.float64)  # distance om M31 centre in units of R25
zT_M31           = np.array(data['zT'].T, dtype=np.float64) # this work z's and errors, incl. our reanalysis of other spectra
zT_err_M31       = np.array(data['zT_err'].T, dtype=np.float64)
zD_M31           = np.array(data['zD'].T, dtype=np.float64)    # DESI collaboration, Dey et AL. z's; there are no individual errors
zQ_M31           = np.array(data['zQ'].T, dtype=np.float64)    # Quaia, Storey-Fisher et al. z's and errors
zQ_err_M31       = np.array(data['zQ_err'].T, dtype=np.float64)
g_mag_M31        = np.array(data['g_mag'].T, dtype=np.float64) # Pan-STARSS apparent magnitudes and errors in grizy bands, if available
g_mag_err_M31    = np.array(data['g_mag_err'].T, dtype=np.float64)
r_mag_M31        = np.array(data['r_mag'].T, dtype=np.float64)
r_mag_err_M31    = np.array(data['r_mag_err'].T, dtype=np.float64)
i_mag_M31        = np.array(data['i_mag'].T, dtype=np.float64)
i_mag_err_M31    = np.array(data['i_mag_err'].T, dtype=np.float64)
z_mag_M31        = np.array(data['z_mag'].T, dtype=np.float64)
z_mag_err_M31    = np.array(data['z_mag_err'].T, dtype=np.float64)
y_mag_M31        = np.array(data['y_mag'].T, dtype=np.float64)
y_mag_err_M31    = np.array(data['y_mag_err'].T, dtype=np.float64)
Av_Dalc_M31      = np.array(data['Av_Dalc'].T, dtype=np.float64)  # absorption and error derived by Dalcanton et al.
Av_Dalc_err_M31  = np.array(data['Av_Dalc_err'].T, dtype=np.float64)
Av_Drain_M31     = np.array(data['Av_Drain'].T, dtype=np.float64)  # absorption and error derived by Drain et al.
Av_Drain_err_M31 = np.array(data['Av_Drain_err'].T, dtype=np.float64)
print('Number of quasars behind = ', len(RA_M31))

No_max_M31 = len(RA_M31)
No_M31 = np.arange(1,No_max_M31+1, dtype=int)
i_blazar = 124
Cond_M31 = No_M31!=i_blazar
print(Cond_M31, len(Cond_M31[Cond_M31]))

Ag = AgAv * Rv * Ebv_M31_0
Ar = ArAv * Rv * Ebv_M31_0
Ai = AiAv * Rv * Ebv_M31_0
Az = AzAv * Rv * Ebv_M31_0
Ay = AyAv * Rv * Ebv_M31_0
Ag_err = AgAv * Rv * Ebv_err_M31_0
Ar_err = ArAv * Rv * Ebv_err_M31_0
Ai_err = AiAv * Rv * Ebv_err_M31_0
Az_err = AzAv * Rv * Ebv_err_M31_0
Ay_err = AyAv * Rv * Ebv_err_M31_0
Egr =  Ag - Ar
Egi =  Ag - Ai
Egz =  Ag - Az
Egy =  Ag - Ay
Egr_err = np.sqrt(Ag_err**2 + Ar_err**2)
Egi_err = np.sqrt(Ag_err**2 + Ai_err**2)
Egz_err = np.sqrt(Ag_err**2 + Az_err**2)
Egy_err = np.sqrt(Ag_err**2 + Ay_err**2)

gr_mag_M31     = g_mag_M31 - r_mag_M31
gr_mag_M31_c   = g_mag_M31 - r_mag_M31 - Egr
gr_mag_err_M31 = np.sqrt(g_mag_err_M31**2 + r_mag_err_M31**2 + Egr_err**2)

gi_mag_M31     = g_mag_M31 - i_mag_M31
gi_mag_M31_c   = g_mag_M31 - i_mag_M31 - Egi
gi_mag_err_M31 = np.sqrt(g_mag_err_M31**2 + i_mag_err_M31**2 + Egi_err**2)

gz_mag_M31     = g_mag_M31 - z_mag_M31
gz_mag_M31_c   = g_mag_M31 - z_mag_M31 - Egz
gz_mag_err_M31 = np.sqrt(g_mag_err_M31**2 + z_mag_err_M31**2 + Egz_err**2)

gy_mag_M31     = g_mag_M31 - y_mag_M31
gy_mag_M31_c   = g_mag_M31 - y_mag_M31 - Egy
gy_mag_err_M31 = np.sqrt(g_mag_err_M31**2 + y_mag_err_M31**2 + Egy_err**2)

No_max_M31 = len(RA_M31)
No_M31 = np.arange(1,No_max_M31+1, dtype=int)

# Selecting the redshifts: first from this work (zT), if not available
# then from Dei et al. (zD) and for the remaining quasars form Quaia (zQ).
z_M31_adopted     = np.where(zT_M31>0,     zT_M31,     -99.9)
z_M31_err_adopted = np.where(zT_err_M31>0, zT_err_M31, -99.9)
print('{:3s}  {:8s} {:8s}  {:8s} {:8s}'.format('No', 'z_adopt','z_err_ad', 'zT','zT_err'))
for l in range(len(No_M31)):
  print('{:3d}  {:8.4f} {:8.4f}  {:8.4f} {:8.4f}'.format(No_M31[l], z_M31_adopted[l],z_M31_err_adopted[l], zT_M31[l],zT_err_M31[l]))
print('Adopting zT :  N_adopted_redshifts = ', len(z_M31_adopted[z_M31_adopted>0]))

zD_err_M31 = np.where(zD_M31>0, 0.005, -9.99)
z_M31_adopted     = np.where(z_M31_adopted>0,     z_M31_adopted,     zD_M31)
z_M31_err_adopted = np.where(z_M31_err_adopted>0, z_M31_err_adopted, zD_err_M31)
print('{:3s}  {:8s} {:8s}  {:8s} {:8s}  {:8s} {:8s}'.format('No', 'z_adopt','z_err_ad', 'zT','zT_err', 'zD','zD_err'))
for l in range(len(No_M31)):
  print('{:3d}  {:8.4f} {:8.4f}  {:8.4f} {:8.4f}  {:8.4f} {:8.4f}'.format(No_M31[l], z_M31_adopted[l],z_M31_err_adopted[l], zT_M31[l],zT_err_M31[l], zD_M31[l],zD_err_M31[l]))
print('Adopting zD :  N_adopted_redshifts = ', len(z_M31_adopted[z_M31_adopted>0]))

z_M31_adopted     = np.where(z_M31_adopted>0,     z_M31_adopted,     zQ_M31)
z_M31_err_adopted = np.where(z_M31_err_adopted>0, z_M31_err_adopted, zQ_err_M31)
print('{:3s}  {:8s} {:8s}  {:8s} {:8s}  {:8s} {:8s}  {:8s} {:8s}'.format('No', 'z_adopt','z_err_ad', 'zT','zT_err', 'zD','zD_err', 'zQ','zQ_err'))
for l in range(len(No_M31)):
  print('{:3d}  {:8.4f} {:8.4f}  {:8.4f} {:8.4f}  {:8.4f} {:8.4f}  {:8.4f} {:8.4f}'.format(No_M31[l], z_M31_adopted[l],z_M31_err_adopted[l], zT_M31[l],zT_err_M31[l], zD_M31[l],zD_err_M31[l], zQ_M31[l],zQ_err_M31[l]))
print('Adopting zQ :  N_adopted_redshifts = ', len(z_M31_adopted[z_M31_adopted>0]))


###############################################################
# Plot the intrinsic colors of reference quasars vs. redshift.
# Intrinsic colors means that they have been corrected to remove
# the Milky Way absorption and reddening according to the
# reddening map of Schlafly et al. (2011) Then, the colors of
# the reference quasars were averaged in different bins to
# inspect for systematic effects.
#
fig = plt.figure(figsize=(12,8))
# setting labels and conditions to ensure photometry in both bands is available
for j in range(1,4+1,1):
  # selecting the color data for the particular panel
  if(j==1):
    color_c, Color_label, Fname = gr_mag_c, '$g-r$', 'gr'
    Cond = (g_mag_err>0) & (r_mag_err>0) & (g_mag_err<Err_max) & (r_mag_err<Err_max)
  if(j==2):
    color_c, Color_label, Fname = gi_mag_c, '$g-i$', 'gi'
    Cond = (g_mag_err>0) & (i_mag_err>0) & (g_mag_err<Err_max) & (i_mag_err<Err_max)
  if(j==3):
    color_c, Color_label, Fname = gz_mag_c, '$g-z$', 'gz'
    Cond = (g_mag_err>0) & (z_mag_err>0) & (g_mag_err<Err_max) & (z_mag_err<Err_max)
  if(j==4):
    color_c, Color_label, Fname = gy_mag_c, '$g-y$', 'gy'
    Cond = (g_mag_err>0) & (y_mag_err>0) & (g_mag_err<Err_max) & (y_mag_err<Err_max)

  # panel set up
  plt.subplot(2,2,j)
  cm = plt.cm.RdBu
  xx, yy, zz = z[Cond], color_c[Cond], Rho_D25[Cond]
  indexes = xx.argsort()
  xxs, yys, zzs = xx[indexes], yy[indexes], zz[indexes]
  plt.scatter(xxs, yys, marker='.', s=50, linewidths=0, c=zzs, cmap=cm)
  cbar = plt.colorbar()
  cbar.set_label(r'$\rho$')

  # color-redshift relation for the entire sample:
  Bins_z = np.arange(0.0, 6.0, 0.2, dtype=float)
  bin_medians, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic='median', bins=Bins_z, range=None)
  bin_std, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic='std', bins=Bins_z, range=None)
  bin_mad, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic=mad, bins=Bins_z, range=None)
  bin_count, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic='count', bins=Bins_z, range=None)
  bin_width = (bin_edges[1] - bin_edges[0])
  bin_centers = bin_edges[1:] - bin_width/2
  plt.errorbar(bin_centers, bin_medians, yerr=1.46*bin_mad, color='k', zorder=2.5, label=r'$\rho$='+str(Bins_rho[0])+'-'+str(Bins_rho[-1])+'\n     N$_{tot}$='+str(len(xx)))
  # saving the intrinsic colors in ascii files; columns: redshift, color, rms, number of objects in each redshift bin
  np.savetxt('color_'+Fname+'0.dat', np.nan_to_num(np.transpose( [bin_centers, bin_medians, 1.46*bin_mad, bin_count] ),nan=-1e9), fmt='%1.4f %1.4f %1.4f %3d')

  # color-redshift relation for the different radial distance intervals, defined in Bins_rho:
  for i in range(len(Bins_rho)-1):
    Cond = (Bins_rho[i]<=Rho_D25)&(Rho_D25<Bins_rho[i+1])
    bin_medians, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic='median', bins=Bins_z, range=None)
    bin_std, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic='std', bins=Bins_z, range=None)
    bin_mad, bin_edges, bin_number = stats.binned_statistic(z[Cond], color_c[Cond], statistic=mad, bins=Bins_z, range=None)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    plt.plot(bin_centers, bin_medians, '-', label=r'$\rho$='+str(Bins_rho[i])+'-'+str(Bins_rho[i+1]))

  # more plotting panels setting up
  plt.legend(ncol=2, loc='lower right', fontsize='small')
  plt.xlim(0.0,3.1)
  if(j==1):
    plt.ylim(-0.8,1.0)
  if(j==2):
    plt.ylim(-1.2,1.6)
  if(j==3):
    plt.ylim(-1.1,2.1)
  if(j==4):
    plt.ylim(-1.1,1.9)
  plt.xlabel('$z$'); plt.ylabel('('+Color_label+')$_0$, mag')

plt.tight_layout()
plt.savefig(Name_plot1); plt.savefig(Name_plot1[:-2]+'df')
plt.show()
plt.close('all')


###############################################################
# cross-corelation with the existing dust maps: derive the
# intrinsic colors for each M31 quasar from the quasars in the
# reference sample with the same redshift, within +/-dZ_max=0.1
# (and derive also the rms that measures the intrinsic variation
# of the color for the quasars at this redhist); do this for all
# colors, convert the excess to Av and average to obtain a final
# Av for each quasar
#
AvEgr = 1.0/(AgAv-ArAv)
AvEgi = 1.0/(AgAv-AiAv)
AvEgz = 1.0/(AgAv-AzAv)
AvEgy = 1.0/(AgAv-AyAv)
print('AgAx : ', AgAv, ArAv, AiAv, AzAv, AyAv)
print('AvExy : ', AvEgr, AvEgi, AvEgz, AvEgy)
Av_min, Av_max = -2.5, 9.0  # plotting range
fig = plt.figure(figsize=(5.75,6))
for im in range(2):
  # selection of a reddening map from the literature
  if (im==0):
    print('Compare with the map of Draine+')
    Av_M31 = Av_Drain_M31
    Av_err_M31 = Av_Drain_err_M31
    Name_inset = 'Draine'
  if (im==1):
    print('Compare with the map of Dalcanton+')
    Av_M31 = Av_Dalc_M31
    Av_err_M31 = Av_Dalc_err_M31
    print(Av_err_M31)
    Name_inset = 'Dalcanton'

  # set up arrays to store results
  y  = np.zeros(4, dtype=float) - 1001
  ye = np.zeros(4, dtype=float) - 1001
  Av_gr   = np.arange(No_max_M31, dtype=np.float64); Av_gr_err = np.arange(No_max_M31, dtype=np.float64)
  Av_gz   = np.arange(No_max_M31, dtype=np.float64); Av_gz_err = np.arange(No_max_M31, dtype=np.float64)
  Av_gi   = np.arange(No_max_M31, dtype=np.float64); Av_gi_err = np.arange(No_max_M31, dtype=np.float64)
  Av_gy   = np.arange(No_max_M31, dtype=np.float64); Av_gy_err = np.arange(No_max_M31, dtype=np.float64)
  Av_mean = np.arange(No_max_M31, dtype=np.float64); Av_error  = np.arange(No_max_M31, dtype=np.float64)
  N_col   = np.arange(No_max_M31, dtype=int);        N_pts     = np.arange(No_max_M31, dtype=int)

  # cycle over each quasar from the M31 sample
  print('...cycle over each quasar from the M31 sample...')
  for i in range(No_max_M31):  # cycle over the quasars behind M31
    print('============================')
    Cond_zi = (z_M31_adopted[i]>=0) & (np.abs(z_M31_adopted[i]-z)<dZ_max) & (z_M31_adopted[i]<3)
    N_pts1 = len(Cond_zi[Cond_zi==True])
    if ((N_pts1>0) & (i != (i_blazar-1))):
      print('color    mean   median  std    1.46*mad  w_mean w_std   len Appar_col err    Excess  err     Av      err')
      for j in range(1,4+1,1):  # cycle over the four colors
        if(j==1):
          # set up colors of quasars by color; for reference quasars apply the redshift selection defined
          # in Cond_zi so they fall within +/-dZ_max from the redshift of the i-th quasar behind M31
          if (Flag_MW_Av):
            color_ref, color_M31 = gr_mag_c[Cond_zi], gr_mag_M31_c
          else:
            color_ref, color_M31 = gr_mag[Cond_zi], gr_mag_M31
          color_err_ref, color_err_M31, AvE, color_label = gr_mag_err[Cond_zi], gr_mag_err_M31, AvEgr, 'g-r'
        if(j==2):
          if (Flag_MW_Av):
            color_ref, color_M31 = gi_mag_c[Cond_zi], gi_mag_M31_c
          else:
            color_ref,color_M31 = gi_mag[Cond_zi], gi_mag_M31
          color_err_ref, color_err_M31, AvE, color_label = gi_mag_err[Cond_zi], gi_mag_err_M31, AvEgi, 'g-i'
        if(j==3):
          if (Flag_MW_Av):
            color_ref, color_M31 = gz_mag_c[Cond_zi], gz_mag_M31_c
          else:
            color_ref, color_M31 = gz_mag[Cond_zi], gz_mag_M31
          color_err_ref, color_err_M31, AvE, color_label = gz_mag_err[Cond_zi], gz_mag_err_M31, AvEgz, 'g-z'
        if(j==4):
          if (Flag_MW_Av):
            color_ref, color_M31 = gy_mag_c[Cond_zi], gy_mag_M31_c
          else:
            color_ref, color_M31 = gy_mag[Cond_zi], gy_mag_M31
          color_err_ref, color_err_M31, AvE, color_label = gy_mag_err[Cond_zi], gy_mag_err_M31, AvEgy, 'g-y'

        # check if there are good measurements of the i-th M31 object in this band, then determine the
        # intrinsic color (and various statistics) of the reference quasars and finally calculate the
        # color excess, Av and Av_err:
        if ((color_M31[i]<99)&(color_err_M31[i]<99)):
          Mean = np.mean(color_ref); Median = np.median(color_ref); Std = np.std(color_ref)
          MAD = 1.46*mad(color_ref); WMean = weighted_avg_and_std(color_ref,color_err_ref**-2)[0]
          WStd = weighted_avg_and_std(color_ref,color_err_ref**-2)[1]; Len = len(color_ref)
          Excess_mag = color_M31[i] - Median; Excess_mag_err = np.sqrt(color_err_M31[i]**2 + MAD**2)
          Av = AvE * Excess_mag; Av_err = AvE * Excess_mag_err
          print('{:6s} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:4d} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format(
            color_label+' :',Mean, Median, Std, MAD, WMean, WStd, Len, color_M31[i], color_err_M31[i], Excess_mag, Excess_mag_err, Av, Av_err))
        else:
          Av = -1001; Av_err = -1001
        # load the derived parameters in arrays
        y[j-1] = Av; ye[j-1] = Av_err
        Av_mean1 = weighted_avg_and_std(y[y > -999],ye[y > -999]**-2)[0];
        Av_err1 = weighted_avg_and_std(y[y > -999],ye[y > -999]**-2)[1];
        N_col1 = len(y[y > -999])

    else:
      Av_mean1 = -1001.0; Av_err1 = -1001.0; N_col1 = 0
    # store the derived absorptions in arrays
    Av_mean[i] = Av_mean1; Av_error[i] = Av_err1; N_col[i] = N_col1; N_pts[i] = N_pts1
    Av_gr[i] = y[0]; Av_gr_err[i] = ye[0]; Av_gi[i] = y[1]; Av_gi_err[i] = ye[1]
    Av_gy[i] = y[2]; Av_gy_err[i] = ye[2]; Av_gz[i] = y[3]; Av_gz_err[i] = ye[3]
    print('{:3s} {:3d} {:7.4f} {:6.4f} {:3d} {:3d} {:7.5f} {:7.5f}'.format('XXX', i+1, Av_mean1, Av_err1, N_col1, N_pts1, z_M31_adopted[i], z_M31_err_adopted[i]))

  Cond = (Av_err_M31 > 0) & (Av_error > 0)   # select quasars with derived absorptions; otherwise it is negative
  # generate the plot
  if (len(Cond[Cond])>0):
    plt.subplot(2,1,im+1)
    plt.grid(zorder=0)
    x  = Av_M31[Cond]
    xe = Av_err_M31[Cond]
    y  = Av_mean[Cond]
    ye = Av_error[Cond]
    xf = np.linspace(np.min([x,y]), np.max([x,y]), 3)
    plt.errorbar(x, y, xerr=xe, yerr=ye, ls='none', fmt='', zorder=2)
    plt.scatter(x, y, s=9.0, c=z_M31_adopted[Cond], cmap='rainbow', zorder=4)
    cbar = plt.colorbar(); cbar.set_label('Redshift', labelpad=9, rotation=270)
    plt.plot([Av_min, Av_max], [Av_min, Av_max], 'k:', label='Slope=1', zorder=0)
    plt.ylabel('A$_V$(quasars,  N='+str(len(x))+'), mag'); plt.xlabel('A$_V^{'+Name_inset+'}$, mag')
    plt.xlim(Av_min, Av_max); plt.ylim(Av_min, Av_max/2)

plt.tight_layout()
plt.savefig(Name_plot2); plt.savefig(Name_plot2[:-2]+'df')
plt.show()
plt.close('all')

###############################################################
# Save an output fits table - identical with the input fits,
# but with new columns for adopted redshift and derived Av (for
# the blazar Av and Av_err are -1001).
#
hdu = fits.BinTableHDU.from_columns([
    fits.Column(name='No',               format='J',   array=No_M31           ),
    fits.Column(name='PS1_ID_M31',       format='A18', array=PS1_ID_M31       ),
    fits.Column(name='Gaia_DR3_ID',      format='A18', array=Gaia_DR3_ID      ),
    fits.Column(name='RA',               format='D',   array=RA_M31           ),
    fits.Column(name='DEC',              format='D',   array=DEC_M31          ),
    fits.Column(name='Rho_D25_M31',      format='E',   array=Rho_D25_M31      ),
    fits.Column(name='zT_M31',           format='E',   array=zT_M31           ),
    fits.Column(name='zT_err_M31',       format='E',   array=zT_err_M31       ),
    fits.Column(name='zD_M31',           format='E',   array=zD_M31           ),
    fits.Column(name='zQ_M31',           format='E',   array=zQ_M31           ),
    fits.Column(name='zQ_err_M31',       format='E',   array=zQ_err_M31       ),
    fits.Column(name='zA_M31',           format='E',   array=z_M31_adopted    ),
    fits.Column(name='zA_err_M31',       format='E',   array=z_M31_err_adopted),
    fits.Column(name='g_mag_M31',        format='E',   array=g_mag_M31        ),
    fits.Column(name='g_mag_err_M31',    format='E',   array=g_mag_err_M31    ),
    fits.Column(name='r_mag_M31',        format='E',   array=r_mag_M31        ),
    fits.Column(name='r_mag_err_M31',    format='E',   array=r_mag_err_M31    ),
    fits.Column(name='i_mag_M31',        format='E',   array=i_mag_M31        ),
    fits.Column(name='i_mag_err_M31',    format='E',   array=i_mag_err_M31    ),
    fits.Column(name='z_mag_M31',        format='E',   array=z_mag_M31        ),
    fits.Column(name='z_mag_err_M31',    format='E',   array=z_mag_err_M31    ),
    fits.Column(name='y_mag_M31',        format='E',   array=y_mag_M31        ),
    fits.Column(name='y_mag_err_M31',    format='E',   array=y_mag_err_M31    ),
    fits.Column(name='Av_Dalc_M31',      format='E',   array=Av_Dalc_M31      ),
    fits.Column(name='Av_Dalc_err_M31',  format='E',   array=Av_Dalc_err_M31  ),
    fits.Column(name='Av_Drain_M31',     format='E',   array=Av_Drain_M31     ),
    fits.Column(name='Av_Drain_err_M31', format='E',   array=Av_Drain_err_M31 ),
    fits.Column(name='Av_T_M31',         format='E',   array=Av_mean          ),
    fits.Column(name='Av_T_err_M31',     format='E',   array=Av_error         )
    ])
hdu.writeto(NameM31QuasarsOut, overwrite=True)

###############################################################
