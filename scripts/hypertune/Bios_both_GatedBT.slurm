#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_both_GatedBT

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Bios_both_GatedBT_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Bios_both_GatedBT_%J.out

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

python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5546052 --exp_id GatedBT_Resampling_y_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5298215 --exp_id GatedBT_Resampling_y_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9912043 --exp_id GatedBT_Resampling_joint_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6137040 --exp_id GatedBT_Resampling_joint_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7737480 --exp_id GatedBT_Resampling_stratified_y_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9073003 --exp_id GatedBT_Resampling_stratified_y_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7484234 --exp_id GatedBT_Resampling_stratified_g_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3748105 --exp_id GatedBT_Resampling_stratified_g_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2462597 --exp_id GatedBT_Resampling_g_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2379483 --exp_id GatedBT_Resampling_g_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8063401 --exp_id GatedBT_Reweighting_y_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8930331 --exp_id GatedBT_Reweighting_y_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7469543 --exp_id GatedBT_Reweighting_joint_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3693491 --exp_id GatedBT_Reweighting_joint_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4691346 --exp_id GatedBT_Reweighting_stratified_y_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2947952 --exp_id GatedBT_Reweighting_stratified_y_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8666884 --exp_id GatedBT_Reweighting_stratified_g_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7910971 --exp_id GatedBT_Reweighting_stratified_g_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6773721 --exp_id GatedBT_Reweighting_g_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6496564 --exp_id GatedBT_Reweighting_g_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 108469 --exp_id GatedBT_Resampling_y_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1285042 --exp_id GatedBT_Resampling_y_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2015452 --exp_id GatedBT_Resampling_joint_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 713336 --exp_id GatedBT_Resampling_joint_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3475779 --exp_id GatedBT_Resampling_stratified_y_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1322308 --exp_id GatedBT_Resampling_stratified_y_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5348827 --exp_id GatedBT_Resampling_stratified_g_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8476470 --exp_id GatedBT_Resampling_stratified_g_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9399527 --exp_id GatedBT_Resampling_g_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6304978 --exp_id GatedBT_Resampling_g_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5445395 --exp_id GatedBT_Reweighting_y_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6947768 --exp_id GatedBT_Reweighting_y_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1165110 --exp_id GatedBT_Reweighting_joint_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6260445 --exp_id GatedBT_Reweighting_joint_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5524257 --exp_id GatedBT_Reweighting_stratified_y_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7271537 --exp_id GatedBT_Reweighting_stratified_y_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9787749 --exp_id GatedBT_Reweighting_stratified_g_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4716461 --exp_id GatedBT_Reweighting_stratified_g_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 135875 --exp_id GatedBT_Reweighting_g_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7501651 --exp_id GatedBT_Reweighting_g_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7277529 --exp_id GatedBT_Resampling_y_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1547950 --exp_id GatedBT_Resampling_y_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 365558 --exp_id GatedBT_Resampling_joint_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6658273 --exp_id GatedBT_Resampling_joint_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5480534 --exp_id GatedBT_Resampling_stratified_y_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7855420 --exp_id GatedBT_Resampling_stratified_y_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6110267 --exp_id GatedBT_Resampling_stratified_g_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5006069 --exp_id GatedBT_Resampling_stratified_g_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2696756 --exp_id GatedBT_Resampling_g_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5082136 --exp_id GatedBT_Resampling_g_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3922775 --exp_id GatedBT_Reweighting_y_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6186087 --exp_id GatedBT_Reweighting_y_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6375509 --exp_id GatedBT_Reweighting_joint_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2338145 --exp_id GatedBT_Reweighting_joint_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2673079 --exp_id GatedBT_Reweighting_stratified_y_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5401012 --exp_id GatedBT_Reweighting_stratified_y_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1934470 --exp_id GatedBT_Reweighting_stratified_g_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2827019 --exp_id GatedBT_Reweighting_stratified_g_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9927886 --exp_id GatedBT_Reweighting_g_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 550395 --exp_id GatedBT_Reweighting_g_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5815088 --exp_id GatedBT_Resampling_y_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4845019 --exp_id GatedBT_Resampling_y_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4816227 --exp_id GatedBT_Resampling_joint_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2401614 --exp_id GatedBT_Resampling_joint_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8373161 --exp_id GatedBT_Resampling_stratified_y_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9219804 --exp_id GatedBT_Resampling_stratified_y_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2234217 --exp_id GatedBT_Resampling_stratified_g_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3528085 --exp_id GatedBT_Resampling_stratified_g_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5186228 --exp_id GatedBT_Resampling_g_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 837027 --exp_id GatedBT_Resampling_g_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8142032 --exp_id GatedBT_Reweighting_y_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1873926 --exp_id GatedBT_Reweighting_y_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5981547 --exp_id GatedBT_Reweighting_joint_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2857338 --exp_id GatedBT_Reweighting_joint_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1526971 --exp_id GatedBT_Reweighting_stratified_y_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3893868 --exp_id GatedBT_Reweighting_stratified_y_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6256405 --exp_id GatedBT_Reweighting_stratified_g_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8487365 --exp_id GatedBT_Reweighting_stratified_g_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 819508 --exp_id GatedBT_Reweighting_g_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2873886 --exp_id GatedBT_Reweighting_g_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8430 --exp_id GatedBT_Resampling_y_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9436227 --exp_id GatedBT_Resampling_y_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6326145 --exp_id GatedBT_Resampling_joint_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7845226 --exp_id GatedBT_Resampling_joint_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2906559 --exp_id GatedBT_Resampling_stratified_y_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 372075 --exp_id GatedBT_Resampling_stratified_y_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1082921 --exp_id GatedBT_Resampling_stratified_g_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1036925 --exp_id GatedBT_Resampling_stratified_g_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1261951 --exp_id GatedBT_Resampling_g_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9354654 --exp_id GatedBT_Resampling_g_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5240465 --exp_id GatedBT_Reweighting_y_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 161529 --exp_id GatedBT_Reweighting_y_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7306144 --exp_id GatedBT_Reweighting_joint_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1925034 --exp_id GatedBT_Reweighting_joint_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5194569 --exp_id GatedBT_Reweighting_stratified_y_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1092595 --exp_id GatedBT_Reweighting_stratified_y_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 516807 --exp_id GatedBT_Reweighting_stratified_g_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1586678 --exp_id GatedBT_Reweighting_stratified_g_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2441631 --exp_id GatedBT_Reweighting_g_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2866587 --exp_id GatedBT_Reweighting_g_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj g
sleep 2
