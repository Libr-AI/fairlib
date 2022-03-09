#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_both_INLP

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Bios_both_INLP_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Bios_both_INLP_%J.out

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

python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9478273 --exp_id INLP_False_None_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8116763 --exp_id INLP_False_None_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_min_acc 0.5
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5329513 --exp_id INLP_False_balanced_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_discriminator_reweighting balanced --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7520680 --exp_id INLP_False_balanced_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_discriminator_reweighting balanced --INLP_min_acc 0.5
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 501490 --exp_id INLP_True_None_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4807265 --exp_id INLP_True_None_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_min_acc 0.5
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3933874 --exp_id INLP_True_balanced_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_discriminator_reweighting balanced --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5649973 --exp_id INLP_True_balanced_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_discriminator_reweighting balanced --INLP_min_acc 0.5
sleep 2
