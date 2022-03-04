#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name=Adv_hypertune

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Adv_hypertune_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Adv_hypertune_%J.out

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

python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8946333 --exp_id Adv_hypertune_0.07943282347242814_1_0_0 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6194918 --exp_id Adv_hypertune_0.07943282347242814_1_0_1 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7319593 --exp_id Adv_hypertune_0.07943282347242814_1_0_2 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6767733 --exp_id Adv_hypertune_0.07943282347242814_1_0_3 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5084837 --exp_id Adv_hypertune_0.07943282347242814_1_0_4 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2903960 --exp_id Adv_hypertune_0.15848931924611143_1_0_0 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7803643 --exp_id Adv_hypertune_0.15848931924611143_1_0_1 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9511904 --exp_id Adv_hypertune_0.15848931924611143_1_0_2 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4018969 --exp_id Adv_hypertune_0.15848931924611143_1_0_3 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7824309 --exp_id Adv_hypertune_0.15848931924611143_1_0_4 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8141005 --exp_id Adv_hypertune_0.19952623149688797_1_0_0 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4063059 --exp_id Adv_hypertune_0.19952623149688797_1_0_1 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5157685 --exp_id Adv_hypertune_0.19952623149688797_1_0_2 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6027685 --exp_id Adv_hypertune_0.19952623149688797_1_0_3 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1517572 --exp_id Adv_hypertune_0.19952623149688797_1_0_4 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1924012 --exp_id Adv_hypertune_0.25118864315095807_1_0_0 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9123516 --exp_id Adv_hypertune_0.25118864315095807_1_0_1 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7977354 --exp_id Adv_hypertune_0.25118864315095807_1_0_2 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3473923 --exp_id Adv_hypertune_0.25118864315095807_1_0_3 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1243268 --exp_id Adv_hypertune_0.25118864315095807_1_0_4 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1008428 --exp_id Adv_hypertune_0.5011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8532183 --exp_id Adv_hypertune_0.5011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6397586 --exp_id Adv_hypertune_0.5011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 176492 --exp_id Adv_hypertune_0.5011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5242924 --exp_id Adv_hypertune_0.5011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 741919 --exp_id Adv_hypertune_0.6309573444801934_1_0_0 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6630678 --exp_id Adv_hypertune_0.6309573444801934_1_0_1 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7853121 --exp_id Adv_hypertune_0.6309573444801934_1_0_2 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2322987 --exp_id Adv_hypertune_0.6309573444801934_1_0_3 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4092206 --exp_id Adv_hypertune_0.6309573444801934_1_0_4 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2014644 --exp_id Adv_hypertune_0.3981071705534973_1_0_0 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 399100 --exp_id Adv_hypertune_0.3981071705534973_1_0_1 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2916637 --exp_id Adv_hypertune_0.3981071705534973_1_0_2 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9106675 --exp_id Adv_hypertune_0.3981071705534973_1_0_3 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 428177 --exp_id Adv_hypertune_0.3981071705534973_1_0_4 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7161272 --exp_id Adv_hypertune_0.31622776601683794_1_0_0 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3729203 --exp_id Adv_hypertune_0.31622776601683794_1_0_1 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6975289 --exp_id Adv_hypertune_0.31622776601683794_1_0_2 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7396729 --exp_id Adv_hypertune_0.31622776601683794_1_0_3 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1345875 --exp_id Adv_hypertune_0.31622776601683794_1_0_4 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3334481 --exp_id Adv_hypertune_0.7943282347242817_1_0_0 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 496931 --exp_id Adv_hypertune_0.7943282347242817_1_0_1 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9905202 --exp_id Adv_hypertune_0.7943282347242817_1_0_2 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3834839 --exp_id Adv_hypertune_0.7943282347242817_1_0_3 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5603785 --exp_id Adv_hypertune_0.7943282347242817_1_0_4 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6026918 --exp_id Adv_hypertune_1.0_1_0_0 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6861344 --exp_id Adv_hypertune_1.0_1_0_1 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6524884 --exp_id Adv_hypertune_1.0_1_0_2 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7526498 --exp_id Adv_hypertune_1.0_1_0_3 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1615017 --exp_id Adv_hypertune_1.0_1_0_4 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4463985 --exp_id Adv_hypertune_1.2589254117941675_1_0_0 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6655780 --exp_id Adv_hypertune_1.2589254117941675_1_0_1 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4196104 --exp_id Adv_hypertune_1.2589254117941675_1_0_2 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6974765 --exp_id Adv_hypertune_1.2589254117941675_1_0_3 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4493672 --exp_id Adv_hypertune_1.2589254117941675_1_0_4 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5676738 --exp_id Adv_hypertune_2.5118864315095824_1_0_0 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7885769 --exp_id Adv_hypertune_2.5118864315095824_1_0_1 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1530450 --exp_id Adv_hypertune_2.5118864315095824_1_0_2 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1256613 --exp_id Adv_hypertune_2.5118864315095824_1_0_3 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 491507 --exp_id Adv_hypertune_2.5118864315095824_1_0_4 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3022825 --exp_id Adv_hypertune_3.1622776601683795_1_0_0 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8112997 --exp_id Adv_hypertune_3.1622776601683795_1_0_1 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8952338 --exp_id Adv_hypertune_3.1622776601683795_1_0_2 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 46022 --exp_id Adv_hypertune_3.1622776601683795_1_0_3 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4605411 --exp_id Adv_hypertune_3.1622776601683795_1_0_4 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5811754 --exp_id Adv_hypertune_5.011872336272725_1_0_0 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1691308 --exp_id Adv_hypertune_5.011872336272725_1_0_1 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 383828 --exp_id Adv_hypertune_5.011872336272725_1_0_2 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4517568 --exp_id Adv_hypertune_5.011872336272725_1_0_3 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3953496 --exp_id Adv_hypertune_5.011872336272725_1_0_4 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1442043 --exp_id Adv_hypertune_6.309573444801936_1_0_0 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4299354 --exp_id Adv_hypertune_6.309573444801936_1_0_1 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4655972 --exp_id Adv_hypertune_6.309573444801936_1_0_2 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5527198 --exp_id Adv_hypertune_6.309573444801936_1_0_3 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9935395 --exp_id Adv_hypertune_6.309573444801936_1_0_4 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4560078 --exp_id Adv_hypertune_7.943282347242821_1_0_0 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3877899 --exp_id Adv_hypertune_7.943282347242821_1_0_1 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6706616 --exp_id Adv_hypertune_7.943282347242821_1_0_2 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4372174 --exp_id Adv_hypertune_7.943282347242821_1_0_3 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4172097 --exp_id Adv_hypertune_7.943282347242821_1_0_4 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6088178 --exp_id Adv_hypertune_10.0_1_0_0 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8853385 --exp_id Adv_hypertune_10.0_1_0_1 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9122886 --exp_id Adv_hypertune_10.0_1_0_2 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9575172 --exp_id Adv_hypertune_10.0_1_0_3 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1817192 --exp_id Adv_hypertune_10.0_1_0_4 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1699649 --exp_id Adv_hypertune_12.589254117941675_1_0_0 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3372814 --exp_id Adv_hypertune_12.589254117941675_1_0_1 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5725932 --exp_id Adv_hypertune_12.589254117941675_1_0_2 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 411144 --exp_id Adv_hypertune_12.589254117941675_1_0_3 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7879690 --exp_id Adv_hypertune_12.589254117941675_1_0_4 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6933090 --exp_id Adv_hypertune_15.848931924611142_1_0_0 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1901123 --exp_id Adv_hypertune_15.848931924611142_1_0_1 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6899 --exp_id Adv_hypertune_15.848931924611142_1_0_2 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9075331 --exp_id Adv_hypertune_15.848931924611142_1_0_3 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7034717 --exp_id Adv_hypertune_15.848931924611142_1_0_4 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8088460 --exp_id Adv_hypertune_19.952623149688808_1_0_0 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 889481 --exp_id Adv_hypertune_19.952623149688808_1_0_1 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8283487 --exp_id Adv_hypertune_19.952623149688808_1_0_2 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2565537 --exp_id Adv_hypertune_19.952623149688808_1_0_3 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8414496 --exp_id Adv_hypertune_19.952623149688808_1_0_4 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2694188 --exp_id Adv_hypertune_25.11886431509582_1_0_0 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1627306 --exp_id Adv_hypertune_25.11886431509582_1_0_1 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6458592 --exp_id Adv_hypertune_25.11886431509582_1_0_2 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4032348 --exp_id Adv_hypertune_25.11886431509582_1_0_3 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5090317 --exp_id Adv_hypertune_25.11886431509582_1_0_4 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7095880 --exp_id Adv_hypertune_31.622776601683793_1_0_0 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6904392 --exp_id Adv_hypertune_31.622776601683793_1_0_1 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7716070 --exp_id Adv_hypertune_31.622776601683793_1_0_2 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4719548 --exp_id Adv_hypertune_31.622776601683793_1_0_3 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3373564 --exp_id Adv_hypertune_31.622776601683793_1_0_4 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6721093 --exp_id Adv_hypertune_0.015848931924611134_1_0_0 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7082843 --exp_id Adv_hypertune_0.015848931924611134_1_0_1 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1078697 --exp_id Adv_hypertune_0.015848931924611134_1_0_2 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6092595 --exp_id Adv_hypertune_0.015848931924611134_1_0_3 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6057002 --exp_id Adv_hypertune_0.015848931924611134_1_0_4 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7339304 --exp_id Adv_hypertune_0.03162277660168379_1_0_0 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2182579 --exp_id Adv_hypertune_0.03162277660168379_1_0_1 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7986401 --exp_id Adv_hypertune_0.03162277660168379_1_0_2 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8254115 --exp_id Adv_hypertune_0.03162277660168379_1_0_3 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4113648 --exp_id Adv_hypertune_0.03162277660168379_1_0_4 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 397309 --exp_id Adv_hypertune_0.039810717055349734_1_0_0 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5909752 --exp_id Adv_hypertune_0.039810717055349734_1_0_1 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4349966 --exp_id Adv_hypertune_0.039810717055349734_1_0_2 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6891352 --exp_id Adv_hypertune_0.039810717055349734_1_0_3 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5175029 --exp_id Adv_hypertune_0.039810717055349734_1_0_4 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9248990 --exp_id Adv_hypertune_39.810717055349734_1_0_0 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6672947 --exp_id Adv_hypertune_39.810717055349734_1_0_1 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2025263 --exp_id Adv_hypertune_39.810717055349734_1_0_2 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2731923 --exp_id Adv_hypertune_39.810717055349734_1_0_3 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6088044 --exp_id Adv_hypertune_39.810717055349734_1_0_4 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7803966 --exp_id Adv_hypertune_3.981071705534973_1_0_0 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 753981 --exp_id Adv_hypertune_3.981071705534973_1_0_1 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4879287 --exp_id Adv_hypertune_3.981071705534973_1_0_2 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7889274 --exp_id Adv_hypertune_3.981071705534973_1_0_3 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9200333 --exp_id Adv_hypertune_3.981071705534973_1_0_4 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4066001 --exp_id Adv_hypertune_50.11872336272725_1_0_0 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2821180 --exp_id Adv_hypertune_50.11872336272725_1_0_1 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5573132 --exp_id Adv_hypertune_50.11872336272725_1_0_2 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1319591 --exp_id Adv_hypertune_50.11872336272725_1_0_3 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 909901 --exp_id Adv_hypertune_50.11872336272725_1_0_4 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3569705 --exp_id Adv_hypertune_0.012589254117941675_1_0_0 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6316691 --exp_id Adv_hypertune_0.012589254117941675_1_0_1 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7556687 --exp_id Adv_hypertune_0.012589254117941675_1_0_2 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 664037 --exp_id Adv_hypertune_0.012589254117941675_1_0_3 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3753015 --exp_id Adv_hypertune_0.012589254117941675_1_0_4 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 353211 --exp_id Adv_hypertune_63.095734448019364_1_0_0 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5527431 --exp_id Adv_hypertune_63.095734448019364_1_0_1 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3443385 --exp_id Adv_hypertune_63.095734448019364_1_0_2 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6444658 --exp_id Adv_hypertune_63.095734448019364_1_0_3 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4985693 --exp_id Adv_hypertune_63.095734448019364_1_0_4 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6173923 --exp_id Adv_hypertune_0.06309573444801933_1_0_0 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3521107 --exp_id Adv_hypertune_0.06309573444801933_1_0_1 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2943607 --exp_id Adv_hypertune_0.06309573444801933_1_0_2 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1877373 --exp_id Adv_hypertune_0.06309573444801933_1_0_3 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2591104 --exp_id Adv_hypertune_0.06309573444801933_1_0_4 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4056429 --exp_id Adv_hypertune_0.1_1_0_0 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8371526 --exp_id Adv_hypertune_0.1_1_0_1 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4171495 --exp_id Adv_hypertune_0.1_1_0_2 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2074464 --exp_id Adv_hypertune_0.1_1_0_3 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4467867 --exp_id Adv_hypertune_0.1_1_0_4 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 493728 --exp_id Adv_hypertune_0.12589254117941676_1_0_0 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2196363 --exp_id Adv_hypertune_0.12589254117941676_1_0_1 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4778365 --exp_id Adv_hypertune_0.12589254117941676_1_0_2 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1678488 --exp_id Adv_hypertune_0.12589254117941676_1_0_3 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6953020 --exp_id Adv_hypertune_0.12589254117941676_1_0_4 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3953340 --exp_id Adv_hypertune_79.43282347242821_1_0_0 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9853472 --exp_id Adv_hypertune_79.43282347242821_1_0_1 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 283901 --exp_id Adv_hypertune_79.43282347242821_1_0_2 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7318294 --exp_id Adv_hypertune_79.43282347242821_1_0_3 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9698140 --exp_id Adv_hypertune_79.43282347242821_1_0_4 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 411069 --exp_id Adv_hypertune_0.0199526231496888_1_0_0 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5456654 --exp_id Adv_hypertune_0.0199526231496888_1_0_1 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4795076 --exp_id Adv_hypertune_0.0199526231496888_1_0_2 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3781859 --exp_id Adv_hypertune_0.0199526231496888_1_0_3 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3788098 --exp_id Adv_hypertune_0.0199526231496888_1_0_4 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8910663 --exp_id Adv_hypertune_0.05011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1363201 --exp_id Adv_hypertune_0.05011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4680399 --exp_id Adv_hypertune_0.05011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9621550 --exp_id Adv_hypertune_0.05011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8354835 --exp_id Adv_hypertune_0.05011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 858655 --exp_id Adv_hypertune_1.584893192461114_1_0_0 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2965486 --exp_id Adv_hypertune_1.584893192461114_1_0_1 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9654261 --exp_id Adv_hypertune_1.584893192461114_1_0_2 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9016408 --exp_id Adv_hypertune_1.584893192461114_1_0_3 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1542708 --exp_id Adv_hypertune_1.584893192461114_1_0_4 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1093628 --exp_id Adv_hypertune_1.9952623149688808_1_0_0 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2019304 --exp_id Adv_hypertune_1.9952623149688808_1_0_1 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3488656 --exp_id Adv_hypertune_1.9952623149688808_1_0_2 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1985062 --exp_id Adv_hypertune_1.9952623149688808_1_0_3 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8610080 --exp_id Adv_hypertune_1.9952623149688808_1_0_4 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3092733 --exp_id Adv_hypertune_100.0_1_0_0 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 426988 --exp_id Adv_hypertune_100.0_1_0_1 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 274765 --exp_id Adv_hypertune_100.0_1_0_2 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5392547 --exp_id Adv_hypertune_100.0_1_0_3 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9055462 --exp_id Adv_hypertune_100.0_1_0_4 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3224360 --exp_id Adv_hypertune_0.01_1_0_0 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1505462 --exp_id Adv_hypertune_0.01_1_0_1 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6003465 --exp_id Adv_hypertune_0.01_1_0_2 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8591185 --exp_id Adv_hypertune_0.01_1_0_3 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6631852 --exp_id Adv_hypertune_0.01_1_0_4 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7984104 --exp_id Adv_hypertune_0.025118864315095794_1_0_0 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8061289 --exp_id Adv_hypertune_0.025118864315095794_1_0_1 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 653093 --exp_id Adv_hypertune_0.025118864315095794_1_0_2 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8547259 --exp_id Adv_hypertune_0.025118864315095794_1_0_3 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2607793 --exp_id Adv_hypertune_0.025118864315095794_1_0_4 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 4 --epochs 20
sleep 2
