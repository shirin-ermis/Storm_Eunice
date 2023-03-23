## choose mem
mem=$1
fdir="/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/m-clim/STORM-TRACKS/EU025/sfc/pf_${mem}"
for line in $fdir/*
do

filename="${line##*/}"
fname="${filename%.*}"
fpath="${line%/*}"

python NJL_regrid-track-merge-postproc.py $line > "STORM-TRACKS-OUT/${fname}.out"

done
