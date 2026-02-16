#!/bin/bash


##python make_simulations_oblation.py
seeds=(2 1 0)
stress_levels=(low high)
hereditary_levels=(false true)


for seed in "${seeds[@]}"; do
  for hereditary in "${hereditary_levels[@]}"; do
    for stress in "${stress_levels[@]}"; do  
      
      echo "Running seed=$seed stress=$stress hereditary=$hereditary"
      
      python Oblation_MCMC.py $seed $stress $hereditary
      
    done
  done
done
