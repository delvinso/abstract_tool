#!/bin/bash
#mkdir /home/dso/Documents/Projects/abstract_tool/_model/cleaned_data/results/

for dat in $(ls /home/dso/Documents/Projects/abstract_tool/_initial_work/cleaned_data_metadata/*tsv | egrep -v 'test')
do
            echo "${dat}"
            echo "{
                \"train_file\" : \"${dat}\",
                \"test_file\": \"\",
                \"epochs\" : 5,
                \"out_dir\": \"results\",
                \"max_len\" : 512,
                \"seed\" : 2020,
                \"train\": true,
                \"test\": false,
                \"clustering_model\": \"\"
            }" > config_sample.json
            cat config_sample.json
            python3 abstract_tool.py --config_dir=config_sample.json
done



