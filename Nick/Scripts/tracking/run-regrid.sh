## cf
find /gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ -path '*/cf/*.nc' -not -path '*req_files*' -not -path '*/test/*' | while read line
do

filename="${line##*/}"
fname="${filename%.*}"
fpath="${line%/*}"

python NJL_regrid-track-merge-postproc.py $line > "out/${fname}.out"

done

## pf, assuming 51 members
find /gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ -path '*/pf/*.nc' -not -path '*req_files*' -not -path '*/test/*' | while read line
do

filename="${line##*/}"
fname="${filename%.*}"
fpath="${line%/*}"

for number in {0..49}
do
ncks -d number,$number -v msl $line "${fpath}/${fname}_mem$((number+1)).nc"
python NJL_regrid-track-merge-postproc.py "${fpath}/${fname}_mem$((number+1)).nc" > "out/${fname}.out"
rm -v "${fpath}/${fname}_mem$((number+1)).nc"
done

done