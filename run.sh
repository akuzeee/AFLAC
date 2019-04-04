#!/bin/zsh

# var definitions
dataset_name='MNISTR'
gpu=0
test_keys=('0' '15' '30' '45' '60' '75')
biased=1
n_checks=50
num_train_d=1
num_train_e=1
valid='train'
expname='_BM'
option=""


tasks=()
start=0
en=9
for test_key in "${test_keys[@]}"
do
    for i in `seq $start $en`
    do
        #===================== CNN =====================
        tasks+=("run.py -o $dataset_name$test_key$biased$expname with meta_cfg.dataset_name=$dataset_name meta_cfg.test_key=$test_key meta_cfg.gpu=$gpu meta_cfg.biased=$biased meta_cfg.validation=$valid meta_cfg.model='DAN_sim' meta_cfg.seed=$i train_cfg.alpha=0")

        alphas=(0.0001 0.001 0.01 0.1 1 10)
        for alpha in "${alphas[@]}"
        do
            #===================== DAN =====================
            tasks+=("run.py -o $dataset_name$test_key$biased$expname with meta_cfg.dataset_name=$dataset_name meta_cfg.test_key=$test_key meta_cfg.gpu=$gpu meta_cfg.biased=$biased meta_cfg.validation=$valid meta_cfg.model='DAN_sim' meta_cfg.seed=$i train_cfg.alpha=$alpha train_cfg.n_checks=$n_checks train_cfg.num_train_d=$num_train_d$option")

            #===================== AFLAC-Abl =====================
            tasks+=("run.py -o $dataset_name$test_key$biased$expname with meta_cfg.dataset_name=$dataset_name meta_cfg.test_key=$test_key meta_cfg.gpu=$gpu meta_cfg.biased=$biased meta_cfg.validation=$valid meta_cfg.model='AFLAC' meta_cfg.seed=$i train_cfg.alpha=$alpha train_cfg.n_checks=$n_checks train_cfg.num_train_d=$num_train_d train_cfg.num_train_e=$num_train_e  train_cfg.p_d=independent_y$option")

            #===================== AFLAC =====================
            tasks+=("run.py -o $dataset_name$test_key$biased$expname with meta_cfg.dataset_name=$dataset_name meta_cfg.test_key=$test_key meta_cfg.gpu=$gpu meta_cfg.biased=$biased meta_cfg.validation=$valid meta_cfg.model='AFLAC' meta_cfg.seed=$i train_cfg.alpha=$alpha train_cfg.n_checks=$n_checks train_cfg.num_train_d=$num_train_d train_cfg.num_train_e=$num_train_e  train_cfg.p_d=dependent_y$option")

            #===================== CIDDG =====================
            tasks+=("run.py -o $dataset_name$test_key$biased$expname with meta_cfg.dataset_name=$dataset_name meta_cfg.test_key=$test_key meta_cfg.gpu=$gpu meta_cfg.biased=$biased meta_cfg.validation=$valid meta_cfg.model='CIDDG' meta_cfg.seed=$i train_cfg.alpha=$alpha train_cfg.n_checks=$n_checks train_cfg.num_train_d=$num_train_d$option")
        done
    done
done


# four parallel job
printf '%s\n' "${tasks[@]}" | xargs -P12 -L1 echo "python"
printf '%s\n' "${tasks[@]}" | xargs -P12 -L1 python
