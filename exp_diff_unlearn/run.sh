#! /bin/bash

shell_dir=$(cd "$(dirname "$0")";pwd)
echo "shell_dir: ${shell_dir}"

# min_max=("mean" "max")
min_max=("mean")
step_epoch=("step")
threshold=(0.2 0.1 0.01)
impair_lr=(0.0001 0.00001)

for mm in "${min_max[@]}"
do
    for se in "${step_epoch[@]}"
    do
        for th in "${threshold[@]}"
	do
            for lr in "${impair_lr[@]}"
    	    do
                cd ${shell_dir}
                work_dir="unlearn_${mm}_${se}_${th}_lr${lr}"
                mkdir ${work_dir}
                cp ./config.json ${work_dir}
                cp ./submit.sh ${work_dir}
            
                cd ${work_dir}
                sed -i "s/{out_dir}/${work_dir}/g" config.json
                sed -i "s/{lr}/${lr}/g" config.json
                sed -i "s/{th}/${th}/g" config.json
                sed -i "s/{mm}/${mm}/g" config.json
                sed -i "s/{se}/${se}/g" config.json

                res=`sbatch ./submit.sh`
                res=(${res})
                task_id=${res[-1]}
                echo "task_id: ${task_id}"
                touch "task_id_${task_id}"
            done
        done
    done
done
