#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Moji_INLP

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Moji_INLP_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Moji_INLP_%J.out

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

python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 224949 --exp_id INLP_False_None_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9335337 --exp_id INLP_False_None_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_min_acc 0.5
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9007938 --exp_id INLP_False_balanced_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_discriminator_reweighting balanced --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3233366 --exp_id INLP_False_balanced_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_discriminator_reweighting balanced --INLP_min_acc 0.5
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6183742 --exp_id INLP_True_None_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7627232 --exp_id INLP_True_None_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_min_acc 0.5
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 941429 --exp_id INLP_True_balanced_0_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_discriminator_reweighting balanced --INLP_min_acc 0
sleep 2
python main.py --project_dir hypertune7 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8994579 --exp_id INLP_True_balanced_0.5_0 --epochs_since_improvement 10 --INLP --INLP_n 300 --INLP_by_class --INLP_discriminator_reweighting balanced --INLP_min_acc 0.5
sleep 2
