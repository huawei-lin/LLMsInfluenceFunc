#! /bin/bash

shell_dir=$(cd "$(dirname "$0")";pwd)
echo "shell_dir: ${shell_dir}"

test_data_num=55
for (( i=0; i<${test_data_num}; i++ ))
do
    cd ${shell_dir}
    work_dir="test_data_${i}"
    mkdir ${work_dir}
    cp ./config.json ${work_dir}
    cp ./submit.sh ${work_dir}

    outdir="",

    cd ${work_dir}
    sed -i "s/{test_id}/${i}/g" config.json
    sed -i "s/{work_dir}/${work_dir}/g" config.json

    res=`sbatch ./submit.sh`
    res=(${res})
    task_id=${res[-1]}
    echo "task_id: ${task_id}"
    touch "task_id_${task_id}"
done


