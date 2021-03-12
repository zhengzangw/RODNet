config=$1
dir=$2
checkpoint=$3
epoch=$4

python tools/test.py --use_noise_channel --config configs/$config --data_dir data/valid1/rod2021 --res_dir results/valid_1/$dir --checkpoint checkpoints/$3/epoch_$4_final.pkl
python transresults.py -i results/valid_1/$dir/ -o submits/valid_1/$dir
python cruw-devkit/scripts/ex_evaluate_rod2021.py submits/valid_1/$dir rod2021_valid/annotations/test
