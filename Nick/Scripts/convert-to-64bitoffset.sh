find "${1}chtype" -name "*restart_ice.nc*" | while read line
do

fname="${line##*/}"
fpath="${line%/*}"

outpath0="${fpath%chtype/*}"
outpath1="${fpath##*chtype/}"
outpath="${outpath0}${outpath1}"

echo "converting ${line} to 64 bit offset, saving to ${outpath}/${fname}"
nccopy -k 64-bit-offset $line "${outpath}/${fname}"
echo "converted"
echo

done
