find /gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/m-clim/STORM-TRACKS/EU025/sfc/ -path "*.grib" | while read line
do

filename="${line##*/}"
fname="${filename%.*}"
fpath="${line%/*}"

# test if file already preprocessed
if test -f "${fpath}/${fname}.nc"
then
echo "already processed ${line}"
continue
fi

# make directory for temporary files
mkdir -p "${fpath}/tmp"
echo "processing ${line}"
# create a rules file for grib_filter & filter
echo "write \"${fpath}/tmp/[referenceDate]_[dataDate]_[perturbationNumber].grib\";" > rules_file
grib_filter rules_file $line
rm -v rules_file
# convert the filtered grib files to netcdf
for gribfile in $fpath/tmp/*.grib
do
gribfilename="${gribfile##*/}"
gribfname="${gribfilename%.*}"
grib_to_netcdf $gribfile -o "${fpath}/tmp/${gribfname}.nc"
rm -v $gribfile
done
rm -v $line
# run python script to merge the filtered netcdf files
# python merge_hDates.py $fname $fpath
# clean up by removing temp directory
# rm -rv "${fpath}/tmp"
echo "processed ${line}"
echo ""

done
