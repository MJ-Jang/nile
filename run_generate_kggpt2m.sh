GPUDEV=$1
MODELSPLIT=$2
DATASET=$3
DATAROOT=$4
DATASPLIT=$5
cmd="CUDA_VISIBLE_DEVICES="$GPUDEV"  python finetune_kglm.py  \
    --do_generate \
    --cache_dir ./cache \
    --output_dir=./saved_lm/kggpt2_m_"$MODELSPLIT"  \
    --model_type=kggpt2 \
    --model_name_or_path=gpt2-medium \
    --block_size 128 \
    --save_steps 6866800 \
    --num_train_epochs 3 \
    --train_data_file=./dataset_"$DATASET"/"$DATAROOT"/train.tsv \
    --eval_data_file=./dataset_"$DATASET"/"$DATAROOT"/"$DATASPLIT".tsv \
    --entity_file_path=./retrieve_knowledge/RotatE_Wn18_512d.txt"

echo $cmd
eval $cmd
