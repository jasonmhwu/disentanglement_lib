#!/bin/bash

even_spread=false
for model_num in {49..51}
do
	if [ "$even_spread" = true ]
	then
		inner_command="
		eval \$(conda shell.bash hook);
		conda activate tf1.5; 
		cd /home/mwu34/disentanglement_lib;
		CUDA_VISIBLE_DEVICES=$((model_num % 2)) python bin/dlib_reproduce --model_num=$model_num --study=double_descent_study --output_directory=output_study_double_descent/$model_num;
		exit;
	"
	else
		inner_command="
		eval \$(conda shell.bash hook);
		conda activate tf1.5; 
		cd /home/mwu34/disentanglement_lib;
		CUDA_VISIBLE_DEVICES=0 python bin/dlib_reproduce --model_num=$model_num --study=double_descent_study --output_directory=output_study_double_descent/$model_num;
		exit;
	"
	fi
	echo screen -dm bash -c \"$inner_command\"
	# screen -dm bash -c '$inner_command'
done
