#!/bin/bash


python 1autofluorescence.py
python 2ABC.py

mkdir FCYeast2_synth/
python make_synth_data 0
for i in {10..50}
    do 
    python make_synth_data.py $i
done


mkdir dilution12
mkdir dilution12/network_perform
mkdir dilution23
mkdir dilution23/network_perform


python 3FC2_training.py 12
python 3FC2_training.py 23

python 4MCMC_real.py
