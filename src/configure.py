import matplotlib.pyplot as plt

ymax = 200
psd_max = 2e2
psd_min = 5e-2

DUmax = 150
psd_max_DU = 1e2
psd_min_DU = 1e-2

# Bandpass (LP)
post_low_freq = 0.3
post_high_freq = 1.5
pre_low_freq = 0.05  # used for prefiltering
pre_high_freq = 3.0  # used for prefiltering

# Bandpass (SP)
post_low_freq_sp = 1.0
post_high_freq_sp = 10.0
pre_low_freq_sp = 0.1  # used for prefiltering
pre_high_freq_sp = 26.0  # used for prefiltering

vmax = psd_max / 2
vmin = psd_min
cmap = plt.get_cmap("jet", lut=300)

num_mv_Apo = 200  # The number of samples used for moving average
font = 22
times = 50  # time width for spectrogram f_lp*time (s)
loc_legend = "lower right"

f_lp = 6.6  # Sampling rate
f_sp = 53.0  # Sampling rate

########## Apollo response curve ##############
LP_t_samp = 1 / f_lp  # sec
# LP_nfft = N # Sample Number of Simulation output
SP_t_samp = 1 / f_sp  # sec
# SP_nfft = N_sp # Sample Number of Simulation output

# Convert Poles and Zeros to Frequency Response (LP Flat)
AF_scale_fac = 5.18524045674722e18
AF_zeros = [
    -9.97000000000000e-04 - 0.00000000000000e00j,
    0.00000000000000e00 + 0.00000000000000e00j,
    0.00000000000000e00 + 0.00000000000000e00j,
    0.00000000000000e00 + 0.00000000000000e00j,
]
AF_poles = [
    -6.28000000000000e-02 + 0.00000000000000e00j,
    -4.76199964295322e01 + 0.00000000000000e00j,
    -5.94194170468653e-02 + 0.00000000000000e00j,
    -3.26855546475662e-01 + -1.74189592000858e-01j,
    -3.26855546475662e-01 + 1.74189592000858e-01j,
    -3.33954437504881e00 - 8.06237332238962e00j,
    -3.33954437504881e00 + 8.06237332238962e00j,
    -3.33954437504881e00 - 8.06237332238962e00j,
    -3.33954437504881e00 + 8.06237332238962e00j,
    -8.06237332238962e00 - 3.33954437504881e00j,
    -8.06237332238962e00 + 3.33954437504881e00j,
    -8.06237332238962e00 - 3.33954437504881e00j,
    -8.06237332238962e00 + 3.33954437504881e00j,
]
# AF_h, AF_f=paz_to_freq_resp(AF_poles, AF_zeros, AF_scale_fac, LP_t_samp, LP_nfft, freq=True)

# Convert Poles and Zeros to Frequency Response (LP Peak)
AP_scale_fac = 5.18524045674722e18
AP_zeros = [
    0.00000000000000e00 + 0.00000000000000e00j,
    0.00000000000000e00 + 0.00000000000000e00j,
    0.00000000000000e00 + 0.00000000000000e00j,
]
AP_poles = [
    -6.28000000000000e-02 + 0.00000000000000e00j,
    -4.77893183981691e01 + 0.00000000000000e00j,
    -2.71405770680666e-01 - 2.84127616186138e00j,
    -2.71405770680666e-01 + 2.84127616186138e00j,
    -3.33954437504881e00 - 8.06237332238962e00j,
    -3.33954437504881e00 + 8.06237332238962e00j,
    -3.33954437504881e00 - 8.06237332238962e00j,
    -3.33954437504881e00 + 8.06237332238962e00j,
    -8.06237332238962e00 - 3.33954437504881e00j,
    -8.06237332238962e00 + 3.33954437504881e00j,
    -8.06237332238962e00 - 3.33954437504881e00j,
    -8.06237332238962e00 + 3.33954437504881e00j,
]
# AP_h, AP_f=paz_to_freq_resp(AP_poles, AP_zeros, AP_scale_fac, LP_t_samp, LP_nfft, freq=True)

# Convert Poles and Zeros to Frequency Response (LP Tidal)
SP_scale_fac = 5.76082444429034e22
SP_zeros = [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
SP_poles = [
    -5.34070751110265e00 + 3.30987324307279e00j,
    -5.34070751110265e00 - 3.30987324307279e00j,
    -0.31416000000000e00 + 0.00000000000000e00j,
    -5.27719065090915e01 + 2.18588393883507e01j,
    -5.27719065090915e01 + 2.18588393883507e01j,
    -5.27719065090915e01 - 2.18588393883507e01j,
    -5.27719065090915e01 - 2.18588393883507e01j,
    -2.18588393883507e01 + 5.27719065090915e01j,
    -2.18588393883507e01 + 5.27719065090915e01j,
    -2.18588393883507e01 - 5.27719065090915e01j,
    -2.18588393883507e01 - 5.27719065090915e01j,
]
# SP_h, SP_f = paz_to_freq_resp(SP_poles, SP_zeros, SP_scale_fac, SP_t_samp, SP_nfft, freq=True)

################## Seismometer Correction #########################
AF_A0 = 1735742164.9349701  # gain
AF_Ds = 3736331684.7959046  # Sensitivity
AP_A0 = 1387789118.8962953  # gain
AP_Ds = 2987333350.251064  # Sensitivity
# TD_A0 = 4.81281409448854E+01
# TD_Ds = ???
SP_A0 = 5695016579508.034  # gain
SP_Ds = 10115553420.896257  # Sensitivity
paz_AF = {"poles": AF_poles, "zeros": AF_zeros, "gain": AF_A0, "sensitivity": AF_Ds}
paz_AP = {"poles": AP_poles, "zeros": AP_zeros, "gain": AP_A0, "sensitivity": AP_Ds}
paz_SP = {"poles": SP_poles, "zeros": SP_zeros, "gain": SP_A0, "sensitivity": SP_Ds}
ß
