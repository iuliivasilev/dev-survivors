#!/bin/bash
for curr_dir in ./build ./dist ./survivors.egg-info 
do
if [ -d $curr_dir ]; then
echo "delete dir $curr_dir "
rm -rf $curr_dir
# else
# echo "create dir $curr_dir "
# mkdir $curr_dir
fi
done 