#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_gender_GatedBT

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Bios_gender_GatedBT_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Bios_gender_GatedBT_%J.out

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

python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3723310 --exp_id GatedBT_Resampling_y_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1292307 --exp_id GatedBT_Resampling_y_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2934554 --exp_id GatedBT_Resampling_joint_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3041179 --exp_id GatedBT_Resampling_joint_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7919162 --exp_id GatedBT_Resampling_stratified_y_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9911336 --exp_id GatedBT_Resampling_stratified_y_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4794288 --exp_id GatedBT_Resampling_stratified_g_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5225805 --exp_id GatedBT_Resampling_stratified_g_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1116097 --exp_id GatedBT_Resampling_g_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7685086 --exp_id GatedBT_Resampling_g_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9439410 --exp_id GatedBT_Reweighting_y_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1213764 --exp_id GatedBT_Reweighting_y_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3469064 --exp_id GatedBT_Reweighting_joint_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3103118 --exp_id GatedBT_Reweighting_joint_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1577152 --exp_id GatedBT_Reweighting_stratified_y_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7652475 --exp_id GatedBT_Reweighting_stratified_y_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2777799 --exp_id GatedBT_Reweighting_stratified_g_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2931562 --exp_id GatedBT_Reweighting_stratified_g_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5086105 --exp_id GatedBT_Reweighting_g_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 248672 --exp_id GatedBT_Reweighting_g_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6475305 --exp_id GatedBT_Resampling_y_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2059365 --exp_id GatedBT_Resampling_y_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5255678 --exp_id GatedBT_Resampling_joint_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2762419 --exp_id GatedBT_Resampling_joint_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7138215 --exp_id GatedBT_Resampling_stratified_y_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7592218 --exp_id GatedBT_Resampling_stratified_y_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1315619 --exp_id GatedBT_Resampling_stratified_g_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4323181 --exp_id GatedBT_Resampling_stratified_g_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 433022 --exp_id GatedBT_Resampling_g_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2913984 --exp_id GatedBT_Resampling_g_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4997582 --exp_id GatedBT_Reweighting_y_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 417174 --exp_id GatedBT_Reweighting_y_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5311488 --exp_id GatedBT_Reweighting_joint_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7115233 --exp_id GatedBT_Reweighting_joint_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6582898 --exp_id GatedBT_Reweighting_stratified_y_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1464329 --exp_id GatedBT_Reweighting_stratified_y_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8655991 --exp_id GatedBT_Reweighting_stratified_g_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8764949 --exp_id GatedBT_Reweighting_stratified_g_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7726890 --exp_id GatedBT_Reweighting_g_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7673786 --exp_id GatedBT_Reweighting_g_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4535732 --exp_id GatedBT_Resampling_y_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4038065 --exp_id GatedBT_Resampling_y_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9556127 --exp_id GatedBT_Resampling_joint_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 339862 --exp_id GatedBT_Resampling_joint_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9911045 --exp_id GatedBT_Resampling_stratified_y_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2732648 --exp_id GatedBT_Resampling_stratified_y_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9750166 --exp_id GatedBT_Resampling_stratified_g_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5830350 --exp_id GatedBT_Resampling_stratified_g_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 473911 --exp_id GatedBT_Resampling_g_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6141308 --exp_id GatedBT_Resampling_g_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1514442 --exp_id GatedBT_Reweighting_y_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3452677 --exp_id GatedBT_Reweighting_y_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4425989 --exp_id GatedBT_Reweighting_joint_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7946757 --exp_id GatedBT_Reweighting_joint_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7754387 --exp_id GatedBT_Reweighting_stratified_y_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8560136 --exp_id GatedBT_Reweighting_stratified_y_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8636818 --exp_id GatedBT_Reweighting_stratified_g_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7400361 --exp_id GatedBT_Reweighting_stratified_g_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5870065 --exp_id GatedBT_Reweighting_g_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2492395 --exp_id GatedBT_Reweighting_g_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2475956 --exp_id GatedBT_Resampling_y_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4379875 --exp_id GatedBT_Resampling_y_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6724091 --exp_id GatedBT_Resampling_joint_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2593615 --exp_id GatedBT_Resampling_joint_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 519395 --exp_id GatedBT_Resampling_stratified_y_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2717343 --exp_id GatedBT_Resampling_stratified_y_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3812588 --exp_id GatedBT_Resampling_stratified_g_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8635200 --exp_id GatedBT_Resampling_stratified_g_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3002163 --exp_id GatedBT_Resampling_g_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2831613 --exp_id GatedBT_Resampling_g_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8359181 --exp_id GatedBT_Reweighting_y_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1267585 --exp_id GatedBT_Reweighting_y_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1212271 --exp_id GatedBT_Reweighting_joint_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7694386 --exp_id GatedBT_Reweighting_joint_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7103700 --exp_id GatedBT_Reweighting_stratified_y_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5800773 --exp_id GatedBT_Reweighting_stratified_y_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8406960 --exp_id GatedBT_Reweighting_stratified_g_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4208717 --exp_id GatedBT_Reweighting_stratified_g_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3335139 --exp_id GatedBT_Reweighting_g_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9146950 --exp_id GatedBT_Reweighting_g_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3764919 --exp_id GatedBT_Resampling_y_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 92460 --exp_id GatedBT_Resampling_y_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4576246 --exp_id GatedBT_Resampling_joint_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8680253 --exp_id GatedBT_Resampling_joint_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8294077 --exp_id GatedBT_Resampling_stratified_y_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9088590 --exp_id GatedBT_Resampling_stratified_y_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4961740 --exp_id GatedBT_Resampling_stratified_g_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2232330 --exp_id GatedBT_Resampling_stratified_g_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9551743 --exp_id GatedBT_Resampling_g_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4904101 --exp_id GatedBT_Resampling_g_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 811367 --exp_id GatedBT_Reweighting_y_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2049340 --exp_id GatedBT_Reweighting_y_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4196450 --exp_id GatedBT_Reweighting_joint_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5411556 --exp_id GatedBT_Reweighting_joint_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7474644 --exp_id GatedBT_Reweighting_stratified_y_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8661357 --exp_id GatedBT_Reweighting_stratified_y_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 833610 --exp_id GatedBT_Reweighting_stratified_g_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9567645 --exp_id GatedBT_Reweighting_stratified_g_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9060904 --exp_id GatedBT_Reweighting_g_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4784756 --exp_id GatedBT_Reweighting_g_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
