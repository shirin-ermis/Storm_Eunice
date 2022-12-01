for scenario in rcp26 rcp85
do

## SET INPUT VARIABLES

# set GCM/RCM combination
GCM=$1 # MPI-M-MPI-ESM-LR NCC-NorESM1-M ICHEC-EC-EARTH
RCM=$2 # HIRHAM5 COSMO-crCLIM-v1-1 COSMO-crCLIM-v1-1

echo "starting cyclone tracking & metric extraction for ${GCM} ${RCM} ${scenario}"

# create script output directories
mkdir -p "out/${GCM}/${RCM}/${scenario}"

# set IO directories
in_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/${scenario}/psl/"
out_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/${scenario}/postproc/cyclonetracking/"

# set regridding parameters
regrid_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/${scenario}/postproc/cyclonetracking/tmp/"
regrid_var="psl"
regrid_grid="/data/ancil/EASE2_N0_25km_Projection.nc"

# create a usable land-sea mask if none exists
ancil_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/ancil/"
python NJL_generate_noISL_lsm.py $ancil_dir > "out/${GCM}/${RCM}/${scenario}/create_lsm.out"

# ancil files
lsm_file="/data/CORDEX/EUR-11/${GCM}/${RCM}/ancil/sftlf*noISL.nc"

# wind speed directory 
wind_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/${scenario}/sfcWind/"

# historical tracks directory
if [ "$scenario" = "evaluation" ]; then
    hist_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/evaluation/postproc/cyclonetracking/"
else
    hist_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/historical/postproc/cyclonetracking/"
fi

# footprint directory
footprint_dir="/data/CORDEX/EUR-11/${GCM}/${RCM}/${scenario}/postproc/footprints/"

## MAIN SCRIPT START

# # run the regridding script
# psl_fpaths=$in_dir"*.nc"
# for line in $psl_fpaths
# do
# fname="${line##*/}"
# regrid_out="${regrid_dir}${fname}"
# if [ ! -f $regrid_out ]; then # line that says only to regrid if the regridded data doesn't already exist
# python NJL_regridder.py $line $regrid_grid $regrid_var $regrid_out > "out/${GCM}/${RCM}/${scenario}/regrid_${fname}.out"
# fi
# done

# # if model timesteps offset from years -> correct
# if [ "$RCM" = "ALADIN63" ] || [ "$RCM" = "RegCM4-6" ]; then
#     python NJL_postproc_tmp_psl_yearoffset.py $regrid_dir > "out/${GCM}/${RCM}/${scenario}/offset.out"
# fi

# # run the cyclone detection & tracking script
# regrid_fpaths=$regrid_dir"psl*.nc"
# for line in $regrid_fpaths
# do
# python C3_CycloneDetection_12_4.py $line $out_dir "${regrid_dir}elevation_grid.nc" $regrid_var > "out/${GCM}/${RCM}/${scenario}/cyclonedetection_${line##*/}.out"
# done

# # run the system detection script
# systemdetection_out="${out_dir}tracking12_4TestTracks/"
# python C3_SystemDetection_12.py $systemdetection_out $in_dir > "out/${GCM}/${RCM}/${scenario}/systemdetection.out"

# # run the system track postprocessing script
# python NJL_postprocessing_to_pandas.py $systemdetection_out > "out/${GCM}/${RCM}/${scenario}/systempost_processing.out"

# # run the metric extraction script
# python NJL_extracting_wind_metrics.py $out_dir $wind_dir $lsm_file > "out/${GCM}/${RCM}/${scenario}/extracting_metrics.out"

# run the footprint selection & extraction script
python NJL_extracting_footprints.py $out_dir $hist_dir $wind_dir $footprint_dir > "out/${GCM}/${RCM}/${scenario}/selecting_footprints.out"

done