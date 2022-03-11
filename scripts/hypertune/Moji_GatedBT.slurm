#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Moji_GatedBT

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Moji_GatedBT_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Moji_GatedBT_%J.out

#SBATCH -q gpgpumse
#SBATCH -p gpgpu
#SBATCH --gres=gpu:1

# load required modules

module load gcccore/10.2.0
module load python/3.8.6
module load fosscuda/2020b
module load pytorch/1.7.1-python-3.8.6
# module load pytorch/1.4.0-python-3.7.4


cd /data/cephfs/punim1421/Fair_NLP_Classification

python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1338241 --exp_id GatedBT_Resampling_y_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1996604 --exp_id GatedBT_Resampling_y_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8828326 --exp_id GatedBT_Resampling_joint_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5164404 --exp_id GatedBT_Resampling_joint_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8075248 --exp_id GatedBT_Resampling_stratified_y_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 803452 --exp_id GatedBT_Resampling_stratified_y_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 598630 --exp_id GatedBT_Resampling_stratified_g_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7382557 --exp_id GatedBT_Resampling_stratified_g_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4962 --exp_id GatedBT_Resampling_g_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5668725 --exp_id GatedBT_Resampling_g_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1089711 --exp_id GatedBT_Reweighting_y_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9852502 --exp_id GatedBT_Reweighting_y_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9401252 --exp_id GatedBT_Reweighting_joint_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3004731 --exp_id GatedBT_Reweighting_joint_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5921721 --exp_id GatedBT_Reweighting_stratified_y_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1094550 --exp_id GatedBT_Reweighting_stratified_y_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1413514 --exp_id GatedBT_Reweighting_stratified_g_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4736706 --exp_id GatedBT_Reweighting_stratified_g_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4303449 --exp_id GatedBT_Reweighting_g_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4607361 --exp_id GatedBT_Reweighting_g_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6301369 --exp_id GatedBT_Resampling_y_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3075097 --exp_id GatedBT_Resampling_y_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9538930 --exp_id GatedBT_Resampling_joint_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6039800 --exp_id GatedBT_Resampling_joint_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9517728 --exp_id GatedBT_Resampling_stratified_y_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5234156 --exp_id GatedBT_Resampling_stratified_y_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 90025 --exp_id GatedBT_Resampling_stratified_g_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4318288 --exp_id GatedBT_Resampling_stratified_g_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3970812 --exp_id GatedBT_Resampling_g_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1032288 --exp_id GatedBT_Resampling_g_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6791709 --exp_id GatedBT_Reweighting_y_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3651326 --exp_id GatedBT_Reweighting_y_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1308624 --exp_id GatedBT_Reweighting_joint_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9734434 --exp_id GatedBT_Reweighting_joint_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5400342 --exp_id GatedBT_Reweighting_stratified_y_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8765104 --exp_id GatedBT_Reweighting_stratified_y_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9122820 --exp_id GatedBT_Reweighting_stratified_g_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1816250 --exp_id GatedBT_Reweighting_stratified_g_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4294666 --exp_id GatedBT_Reweighting_g_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4438795 --exp_id GatedBT_Reweighting_g_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1856437 --exp_id GatedBT_Resampling_y_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5994465 --exp_id GatedBT_Resampling_y_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1321663 --exp_id GatedBT_Resampling_joint_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7859069 --exp_id GatedBT_Resampling_joint_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9115574 --exp_id GatedBT_Resampling_stratified_y_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9585981 --exp_id GatedBT_Resampling_stratified_y_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 900926 --exp_id GatedBT_Resampling_stratified_g_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1641934 --exp_id GatedBT_Resampling_stratified_g_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3330411 --exp_id GatedBT_Resampling_g_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3861377 --exp_id GatedBT_Resampling_g_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3994559 --exp_id GatedBT_Reweighting_y_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9085932 --exp_id GatedBT_Reweighting_y_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6356668 --exp_id GatedBT_Reweighting_joint_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5729348 --exp_id GatedBT_Reweighting_joint_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3022702 --exp_id GatedBT_Reweighting_stratified_y_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3423957 --exp_id GatedBT_Reweighting_stratified_y_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9357942 --exp_id GatedBT_Reweighting_stratified_g_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6200618 --exp_id GatedBT_Reweighting_stratified_g_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5746184 --exp_id GatedBT_Reweighting_g_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3438025 --exp_id GatedBT_Reweighting_g_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9732918 --exp_id GatedBT_Resampling_y_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5082545 --exp_id GatedBT_Resampling_y_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9800390 --exp_id GatedBT_Resampling_joint_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3809103 --exp_id GatedBT_Resampling_joint_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6416478 --exp_id GatedBT_Resampling_stratified_y_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3131322 --exp_id GatedBT_Resampling_stratified_y_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3397083 --exp_id GatedBT_Resampling_stratified_g_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9643222 --exp_id GatedBT_Resampling_stratified_g_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8974729 --exp_id GatedBT_Resampling_g_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8569339 --exp_id GatedBT_Resampling_g_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2380548 --exp_id GatedBT_Reweighting_y_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1519000 --exp_id GatedBT_Reweighting_y_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 442461 --exp_id GatedBT_Reweighting_joint_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4583461 --exp_id GatedBT_Reweighting_joint_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9619287 --exp_id GatedBT_Reweighting_stratified_y_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5082293 --exp_id GatedBT_Reweighting_stratified_y_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8863510 --exp_id GatedBT_Reweighting_stratified_g_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4678007 --exp_id GatedBT_Reweighting_stratified_g_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9052940 --exp_id GatedBT_Reweighting_g_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3095572 --exp_id GatedBT_Reweighting_g_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2037977 --exp_id GatedBT_Resampling_y_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6213116 --exp_id GatedBT_Resampling_y_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6531200 --exp_id GatedBT_Resampling_joint_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1373615 --exp_id GatedBT_Resampling_joint_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1669373 --exp_id GatedBT_Resampling_stratified_y_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 919085 --exp_id GatedBT_Resampling_stratified_y_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3551850 --exp_id GatedBT_Resampling_stratified_g_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 268510 --exp_id GatedBT_Resampling_stratified_g_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7504829 --exp_id GatedBT_Resampling_g_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4738474 --exp_id GatedBT_Resampling_g_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9520439 --exp_id GatedBT_Reweighting_y_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2366958 --exp_id GatedBT_Reweighting_y_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5984365 --exp_id GatedBT_Reweighting_joint_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2375765 --exp_id GatedBT_Reweighting_joint_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4552150 --exp_id GatedBT_Reweighting_stratified_y_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6701897 --exp_id GatedBT_Reweighting_stratified_y_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6906986 --exp_id GatedBT_Reweighting_stratified_g_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7652143 --exp_id GatedBT_Reweighting_stratified_g_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2192957 --exp_id GatedBT_Reweighting_g_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9739924 --exp_id GatedBT_Reweighting_g_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
