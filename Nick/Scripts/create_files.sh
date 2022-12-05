# CONFIG
out_dir="/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/output/"
for inidate in 20220213
do
echo "computing perturbed restarts for ${inidate}"
restart_dir="/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/restarts/${inidate}/"
gridfile="/home/l/leach/Attribution/NA-Storms/Ancil/Tco_639.grib"

# CREATE DIRECTORIES
mkdir -p "${out_dir}/tmp"
mkdir -p $out_dir/{pi,incr}-co2/{0001/opa{0..4}/restart/2022,sstdelta}

# mkdir -p $out_dir/chtype/{pi,incr}-co2/0001/opa{0..4}/restart/2022

# RUN PYTHON SCRIPTS
python "generate-perturbation.py" $out_dir
python "perturb-restarts.py" $out_dir $restart_dir
for opa in opa{0..4}
do
for perturbation in pi-co2 incr-co2
do
python "modify-salinity.py" $out_dir $restart_dir $opa $perturbation
done
done

# IF SEAS5 RESTARTS, CONVERT TO 64 BIT OFFSET
# bash convert-to-64bitoffset.sh $out_dir

# MOVE PERTURBED RESTARTS TO CORRECT DIRECTORY
cp $out_dir/tmp/t2d_remap.nc $out_dir/tmp/sstdelta.nc
# for fname in $out_dir/tmp/*_restart_ice.nc
# do
# IFS="/" read -r -a fname_arr <<< $fname
# IFS="-" read -r -a fname_arr1 <<< "${fname_arr[-1]}"
# IFS="_" read -r -a fname_arr2 <<< "${fname_arr1[-1]}"
# cp $fname "${out_dir}/${fname_arr1[1]}-${fname_arr1[2]}/${fname_arr2[0]}/${fname_arr1[0]}/restart/2022/${fname_arr1[-1]}"
# done


# GENERATE sstdelta grib
## need the cdo conda env
# conda activate cdo_env
## create grid description file from example Tco639 gridded file
cdo griddes $gridfile > $out_dir/tmp/griddes
## copy to grib file, setting land values to zero
cdo -f grb copy -setmissval,-1e20 -setctomiss,nan -remapbil,$out_dir/tmp/griddes $out_dir/tmp/sstdelta.nc $out_dir/tmp/incr-co2-sstdelta-remap.grib
## same + *-1 for pi-co2 perturbation
cdo -f grb copy -setmissval,-1e20 -setctomiss,nan -mulc,-1 -remapbil,$out_dir/tmp/griddes $out_dir/tmp/sstdelta.nc $out_dir/tmp/pi-co2-sstdelta-remap.grib

## set some required grib keys to allow mars to read
for perturbation in pi-co2 incr-co2
do
   grib_set -s centre:s=ecmf,indicatorOfParameter=161,table2Version=151,indicatorOfTypeOfLevel=1,level=0,dataDate=$inidate,dataTime=0 "${out_dir}/tmp/${perturbation}-sstdelta-remap.grib" "${out_dir}/${perturbation}/sstdelta/sstdelta.grib"
done

rm "${out_dir}/tmp/*"
echo "compted perturbed restarts for ${inidate} and cleaned /tmp/"

done
## return to base env
# conda deactivate
