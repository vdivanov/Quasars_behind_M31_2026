
# Quasars behind M31

This repository contains the data used in Nedialkov et al. (2026):


(1) Input fits tables:

-- quasars_M31.fits with the data for the quasars behind M31 and

-- quasars_ref.fits with the data for the reference sample of quasars outside M31.


(2) The repository contains a stand-alone script that calculates:

-- the dereddened intrinsic colors of quasars as a function of redshift and

-- the color excess and absorption for each of the quasars behind M31.

The script also generates two of the plots of the paper.


The output includes:

-- quasars_M31_out.fits - a fits table with the data for the quasars behind M31, with new columns containing for each quasar the adopted redshift and the derived absorption

-- color_xx0.dat where xx mark the color - ascii files containing the intrinsic dereddened colors as a function of redshift

-- colors_dered.* and Av_quasars_vs_Av_maps.* - plots of the intrinsic dereddened quasar colors as a function of redshift


## Usage

python3 color_excesses.py
