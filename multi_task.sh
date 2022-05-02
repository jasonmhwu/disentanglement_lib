#!/bin/bash

even_spread=true
study_name=embeddings_study_v3
pipeline_name=dlib_reproduce_semi_supervised
sleep_command='sleep 11000'
only_train=false

if [ "$only_train" = true ]
then
	only_train_flag=--only_train
else
	only_train_flag=''
fi

for model_num in {9..17}
do
	if [ "$even_spread" = true ]
	then
		inner_command="
		eval \$(conda shell.bash hook);
		conda activate tf1.5; 
		cd /home/mwu34/disentanglement_lib;
		$sleep_command;
		CUDA_VISIBLE_DEVICES=$((model_num % 2)) python bin/$pipeline_name --model_num=$model_num --study=$study_name --output_directory=output_$study_name/$model_num $only_train_flag;
		exit;
	"
	else
		inner_command="
		eval \$(conda shell.bash hook);
		conda activate tf1.5; 
		cd /home/mwu34/disentanglement_lib;
		$sleep_command;
		CUDA_VISIBLE_DEVICES=0 python bin/$pipeline_name --model_num=$model_num --study=$study_name --output_directory=output_$study_name/$model_num $only_train_flag;
		exit;
	"
	fi
	echo screen -dm bash -c \"$inner_command\"
	# screen -dm bash -c '$inner_command'
done
