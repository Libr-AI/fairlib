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

python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3607475 --exp_id GatedBT_Reweighting_stratified_g__0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7257081 --exp_id GatedBT_Reweighting_stratified_g_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4129363 --exp_id GatedBT_Reweighting_stratified_y__0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7053958 --exp_id GatedBT_Reweighting_stratified_y_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6445890 --exp_id GatedBT_Reweighting_y__0 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2869734 --exp_id GatedBT_Reweighting_y_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7057620 --exp_id GatedBT_Reweighting_g__0 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2615130 --exp_id GatedBT_Reweighting_g_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4182601 --exp_id GatedBT_Reweighting_joint__0 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5145114 --exp_id GatedBT_Reweighting_joint_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4718690 --exp_id GatedBT_Resampling_stratified_g__0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8638893 --exp_id GatedBT_Resampling_stratified_g_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5527 --exp_id GatedBT_Resampling_stratified_y__0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1476458 --exp_id GatedBT_Resampling_stratified_y_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8435448 --exp_id GatedBT_Resampling_y__0 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2723103 --exp_id GatedBT_Resampling_y_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8414292 --exp_id GatedBT_Resampling_g__0 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5001987 --exp_id GatedBT_Resampling_g_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4183232 --exp_id GatedBT_Resampling_joint__0 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7277943 --exp_id GatedBT_Resampling_joint_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 198876 --exp_id GatedBT_Reweighting_stratified_g__1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4751337 --exp_id GatedBT_Reweighting_stratified_g_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4418354 --exp_id GatedBT_Reweighting_stratified_y__1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6930139 --exp_id GatedBT_Reweighting_stratified_y_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 635402 --exp_id GatedBT_Reweighting_y__1 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6539295 --exp_id GatedBT_Reweighting_y_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2861275 --exp_id GatedBT_Reweighting_g__1 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4982302 --exp_id GatedBT_Reweighting_g_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9672283 --exp_id GatedBT_Reweighting_joint__1 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8760372 --exp_id GatedBT_Reweighting_joint_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1496002 --exp_id GatedBT_Resampling_stratified_g__1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3847189 --exp_id GatedBT_Resampling_stratified_g_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9097456 --exp_id GatedBT_Resampling_stratified_y__1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2935071 --exp_id GatedBT_Resampling_stratified_y_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8866841 --exp_id GatedBT_Resampling_y__1 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6614311 --exp_id GatedBT_Resampling_y_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6788851 --exp_id GatedBT_Resampling_g__1 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2723546 --exp_id GatedBT_Resampling_g_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7238008 --exp_id GatedBT_Resampling_joint__1 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3941809 --exp_id GatedBT_Resampling_joint_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 115981 --exp_id GatedBT_Reweighting_stratified_g__2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 608772 --exp_id GatedBT_Reweighting_stratified_g_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7591271 --exp_id GatedBT_Reweighting_stratified_y__2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8842184 --exp_id GatedBT_Reweighting_stratified_y_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7407357 --exp_id GatedBT_Reweighting_y__2 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2637774 --exp_id GatedBT_Reweighting_y_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2215422 --exp_id GatedBT_Reweighting_g__2 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1161007 --exp_id GatedBT_Reweighting_g_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9250503 --exp_id GatedBT_Reweighting_joint__2 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5343818 --exp_id GatedBT_Reweighting_joint_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6559965 --exp_id GatedBT_Resampling_stratified_g__2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2381166 --exp_id GatedBT_Resampling_stratified_g_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 546429 --exp_id GatedBT_Resampling_stratified_y__2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7257919 --exp_id GatedBT_Resampling_stratified_y_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5691165 --exp_id GatedBT_Resampling_y__2 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6258159 --exp_id GatedBT_Resampling_y_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5646165 --exp_id GatedBT_Resampling_g__2 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1785485 --exp_id GatedBT_Resampling_g_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1920112 --exp_id GatedBT_Resampling_joint__2 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9270145 --exp_id GatedBT_Resampling_joint_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 69945 --exp_id GatedBT_Reweighting_stratified_g__3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4859977 --exp_id GatedBT_Reweighting_stratified_g_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 643503 --exp_id GatedBT_Reweighting_stratified_y__3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9175032 --exp_id GatedBT_Reweighting_stratified_y_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8599928 --exp_id GatedBT_Reweighting_y__3 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9519707 --exp_id GatedBT_Reweighting_y_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8579201 --exp_id GatedBT_Reweighting_g__3 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6222462 --exp_id GatedBT_Reweighting_g_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7499472 --exp_id GatedBT_Reweighting_joint__3 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9758983 --exp_id GatedBT_Reweighting_joint_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6560910 --exp_id GatedBT_Resampling_stratified_g__3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 605865 --exp_id GatedBT_Resampling_stratified_g_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2259018 --exp_id GatedBT_Resampling_stratified_y__3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5528784 --exp_id GatedBT_Resampling_stratified_y_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3029601 --exp_id GatedBT_Resampling_y__3 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 658756 --exp_id GatedBT_Resampling_y_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7156123 --exp_id GatedBT_Resampling_g__3 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5247766 --exp_id GatedBT_Resampling_g_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3687298 --exp_id GatedBT_Resampling_joint__3 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5589475 --exp_id GatedBT_Resampling_joint_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7281820 --exp_id GatedBT_Reweighting_stratified_g__4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2756797 --exp_id GatedBT_Reweighting_stratified_g_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1074399 --exp_id GatedBT_Reweighting_stratified_y__4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6555965 --exp_id GatedBT_Reweighting_stratified_y_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7362060 --exp_id GatedBT_Reweighting_y__4 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2857910 --exp_id GatedBT_Reweighting_y_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 282900 --exp_id GatedBT_Reweighting_g__4 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7612292 --exp_id GatedBT_Reweighting_g_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6857578 --exp_id GatedBT_Reweighting_joint__4 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6941939 --exp_id GatedBT_Reweighting_joint_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 482565 --exp_id GatedBT_Resampling_stratified_g__4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9452914 --exp_id GatedBT_Resampling_stratified_g_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7076885 --exp_id GatedBT_Resampling_stratified_y__4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2500707 --exp_id GatedBT_Resampling_stratified_y_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5808060 --exp_id GatedBT_Resampling_y__4 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3123086 --exp_id GatedBT_Resampling_y_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6993275 --exp_id GatedBT_Resampling_g__4 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5975560 --exp_id GatedBT_Resampling_g_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6915877 --exp_id GatedBT_Resampling_joint__4 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6499349 --exp_id GatedBT_Resampling_joint_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
