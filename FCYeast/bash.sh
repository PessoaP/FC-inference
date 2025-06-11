#!/bin/bash

mkdir FCYeast_synth/
mkdir FCYeast_MCMC/
python make_synth_data.py 0
for i in {10..50}
    do 
    python make_synth_data.py $i
done