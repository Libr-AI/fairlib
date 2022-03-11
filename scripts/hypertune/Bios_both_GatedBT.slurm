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

python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7730209 --exp_id GatedBT_Reweighting_stratified_g__0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6933178 --exp_id GatedBT_Reweighting_stratified_g_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3450234 --exp_id GatedBT_Reweighting_stratified_y__0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2985067 --exp_id GatedBT_Reweighting_stratified_y_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1287719 --exp_id GatedBT_Reweighting_y__0 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7065469 --exp_id GatedBT_Reweighting_y_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6571833 --exp_id GatedBT_Reweighting_g__0 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2779926 --exp_id GatedBT_Reweighting_g_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4424619 --exp_id GatedBT_Reweighting_joint__0 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2270680 --exp_id GatedBT_Reweighting_joint_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6975243 --exp_id GatedBT_Resampling_stratified_g__0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 379977 --exp_id GatedBT_Resampling_stratified_g_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7978003 --exp_id GatedBT_Resampling_stratified_y__0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4576997 --exp_id GatedBT_Resampling_stratified_y_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5752911 --exp_id GatedBT_Resampling_y__0 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1539606 --exp_id GatedBT_Resampling_y_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2312548 --exp_id GatedBT_Resampling_g__0 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3180688 --exp_id GatedBT_Resampling_g_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2981916 --exp_id GatedBT_Resampling_joint__0 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9350013 --exp_id GatedBT_Resampling_joint_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7604475 --exp_id GatedBT_Reweighting_stratified_g__1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 392590 --exp_id GatedBT_Reweighting_stratified_g_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1544524 --exp_id GatedBT_Reweighting_stratified_y__1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 386660 --exp_id GatedBT_Reweighting_stratified_y_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1326947 --exp_id GatedBT_Reweighting_y__1 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8662710 --exp_id GatedBT_Reweighting_y_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3543826 --exp_id GatedBT_Reweighting_g__1 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5107985 --exp_id GatedBT_Reweighting_g_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9343377 --exp_id GatedBT_Reweighting_joint__1 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6441561 --exp_id GatedBT_Reweighting_joint_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 193680 --exp_id GatedBT_Resampling_stratified_g__1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7721599 --exp_id GatedBT_Resampling_stratified_g_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2825384 --exp_id GatedBT_Resampling_stratified_y__1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 898144 --exp_id GatedBT_Resampling_stratified_y_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3663063 --exp_id GatedBT_Resampling_y__1 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5733133 --exp_id GatedBT_Resampling_y_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3812319 --exp_id GatedBT_Resampling_g__1 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5782115 --exp_id GatedBT_Resampling_g_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6506553 --exp_id GatedBT_Resampling_joint__1 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5842671 --exp_id GatedBT_Resampling_joint_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5463366 --exp_id GatedBT_Reweighting_stratified_g__2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1610355 --exp_id GatedBT_Reweighting_stratified_g_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1843753 --exp_id GatedBT_Reweighting_stratified_y__2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4431088 --exp_id GatedBT_Reweighting_stratified_y_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5453486 --exp_id GatedBT_Reweighting_y__2 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6134831 --exp_id GatedBT_Reweighting_y_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2239292 --exp_id GatedBT_Reweighting_g__2 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7863413 --exp_id GatedBT_Reweighting_g_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5835415 --exp_id GatedBT_Reweighting_joint__2 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3117976 --exp_id GatedBT_Reweighting_joint_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6947080 --exp_id GatedBT_Resampling_stratified_g__2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3112039 --exp_id GatedBT_Resampling_stratified_g_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8706692 --exp_id GatedBT_Resampling_stratified_y__2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9802668 --exp_id GatedBT_Resampling_stratified_y_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8274145 --exp_id GatedBT_Resampling_y__2 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5566686 --exp_id GatedBT_Resampling_y_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4190986 --exp_id GatedBT_Resampling_g__2 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4656872 --exp_id GatedBT_Resampling_g_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3569962 --exp_id GatedBT_Resampling_joint__2 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4394757 --exp_id GatedBT_Resampling_joint_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7885191 --exp_id GatedBT_Reweighting_stratified_g__3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9331236 --exp_id GatedBT_Reweighting_stratified_g_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6073992 --exp_id GatedBT_Reweighting_stratified_y__3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9322415 --exp_id GatedBT_Reweighting_stratified_y_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9452780 --exp_id GatedBT_Reweighting_y__3 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7133216 --exp_id GatedBT_Reweighting_y_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1878511 --exp_id GatedBT_Reweighting_g__3 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8733778 --exp_id GatedBT_Reweighting_g_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6718496 --exp_id GatedBT_Reweighting_joint__3 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3669324 --exp_id GatedBT_Reweighting_joint_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7079350 --exp_id GatedBT_Resampling_stratified_g__3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9203095 --exp_id GatedBT_Resampling_stratified_g_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 951882 --exp_id GatedBT_Resampling_stratified_y__3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3801548 --exp_id GatedBT_Resampling_stratified_y_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 166137 --exp_id GatedBT_Resampling_y__3 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7612958 --exp_id GatedBT_Resampling_y_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6995547 --exp_id GatedBT_Resampling_g__3 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9764956 --exp_id GatedBT_Resampling_g_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 204449 --exp_id GatedBT_Resampling_joint__3 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7404406 --exp_id GatedBT_Resampling_joint_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4901866 --exp_id GatedBT_Reweighting_stratified_g__4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9148282 --exp_id GatedBT_Reweighting_stratified_g_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4393546 --exp_id GatedBT_Reweighting_stratified_y__4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5107526 --exp_id GatedBT_Reweighting_stratified_y_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9376061 --exp_id GatedBT_Reweighting_y__4 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 658126 --exp_id GatedBT_Reweighting_y_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8273565 --exp_id GatedBT_Reweighting_g__4 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4969544 --exp_id GatedBT_Reweighting_g_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3191415 --exp_id GatedBT_Reweighting_joint__4 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4914617 --exp_id GatedBT_Reweighting_joint_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4111915 --exp_id GatedBT_Resampling_stratified_g__4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 690275 --exp_id GatedBT_Resampling_stratified_g_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8703610 --exp_id GatedBT_Resampling_stratified_y__4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 798378 --exp_id GatedBT_Resampling_stratified_y_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3383140 --exp_id GatedBT_Resampling_y__4 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9732864 --exp_id GatedBT_Resampling_y_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 935937 --exp_id GatedBT_Resampling_g__4 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1170424 --exp_id GatedBT_Resampling_g_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 22226 --exp_id GatedBT_Resampling_joint__4 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1457555 --exp_id GatedBT_Resampling_joint_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
