#!/bin/bash

targeteds="True False"
norms="inf two"
iterations="3 5 10 20 40 100"
alphas="0.01"
epsilons="0.03"
attacks="cospgd pgd segpgd"

for attack in $attacks
do
    for iteration in $iterations
    do
        for targeted in $targeteds
        do
            for norm in $norms
            do
                if [[ $norm = "two" ]]
                then
                    alphas="0.1 0.2"
                    epsilons="0.251 0.502"
                else
                    alphas="0.01"
                    epsilons="0.03"
                fi
                for alpha in $alphas
                do
                    for epsilon in $epsilons
                    do
                        job_name="${attack}_its_${iteration}_target_${targeted}_l_${norm}_a_${alpha}_eps_${epsilon}"
                        out_dir="$slurm/{job_name}.out"
                        err_dir="$slurm/{job_name}.err"
                        sbatch -J $job_name --output=out_dir --error=err_dir attack.sh $attack $iteration $targeted $norm $alpha $epsilon
                    done
                done
            done
        done
    done
done