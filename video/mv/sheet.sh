#!/bin/bash
# sheet.sh dir out.png [cols] : tile all pngs in dir (sorted) into a contact sheet
dir=$1; out=$2; cols=${3:-2}
n=$(ls $dir/t*.png | wc -l); rows=$(( (n + cols - 1) / cols ))
ffmpeg -v error -y -pattern_type glob -i "$dir/t*.png" -vf "scale=960:540,drawtext=text='%{metadata\:lavf.image2dec.source_basename}':x=10:y=10:fontsize=22:fontcolor=yellow:box=1:boxcolor=black,tile=${cols}x${rows}" -frames:v 1 $out 2>/dev/null || \
ffmpeg -v error -y -pattern_type glob -i "$dir/t*.png" -vf "scale=960:540,tile=${cols}x${rows}" -frames:v 1 $out
