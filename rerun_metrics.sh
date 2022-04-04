#!/bin/bash

even_spread=false
study_name=active_learning_study_v9
pipeline_name=dlib_reproduce_semi_supervised
for model_num in {41..42}
do
	if [ "$even_spread" = true ]
	then
		inner_command="
		eval \$(conda shell.bash hook);
		conda activate tf1.5; 
		cd /home/mwu34/disentanglement_lib;
		CUDA_VISIBLE_DEVICES=$((model_num % 2)) python bin/$pipeline_name --model_num=$model_num --study=$study_name --output_directory=output_$study_name/$model_num --model_dir=output_$study_name/$model_num/model;
		exit;
	"
	else
		inner_command="
		eval \$(conda shell.bash hook);
		conda activate tf1.5; 
		cd /home/mwu34/disentanglement_lib;
		CUDA_VISIBLE_DEVICES='' python bin/$pipeline_name --model_num=$model_num --study=$study_name --output_directory=output_$study_name/$model_num --model_dir=output_$study_name/$model_num/model;
		exit;
	"
	fi
	echo screen -dm bash -c \"$inner_command\"
	# screen -dm bash -c '$inner_command'
done
