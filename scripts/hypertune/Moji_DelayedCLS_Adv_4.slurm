#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name=DelayedCLS_Adv_4

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_4_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_4_%J.out

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

python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9581 --exp_id DelayedCLS_Adv_4_0.00630957344480193_4_2017 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5101115 --exp_id DelayedCLS_Adv_4_0.00630957344480193_4_2020 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3275481 --exp_id DelayedCLS_Adv_4_0.00630957344480193_4_2022 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5541613 --exp_id DelayedCLS_Adv_4_0.00630957344480193_4_2024 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9586031 --exp_id DelayedCLS_Adv_4_0.00630957344480193_4_2013 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2018491 --exp_id DelayedCLS_Adv_4_0.15848931924611143_4_2017 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4729613 --exp_id DelayedCLS_Adv_4_0.15848931924611143_4_2020 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6724528 --exp_id DelayedCLS_Adv_4_0.15848931924611143_4_2022 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 620536 --exp_id DelayedCLS_Adv_4_0.15848931924611143_4_2024 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5662750 --exp_id DelayedCLS_Adv_4_0.15848931924611143_4_2013 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 267333 --exp_id DelayedCLS_Adv_4_0.19952623149688808_4_2017 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 309656 --exp_id DelayedCLS_Adv_4_0.19952623149688808_4_2020 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3647441 --exp_id DelayedCLS_Adv_4_0.19952623149688808_4_2022 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4080360 --exp_id DelayedCLS_Adv_4_0.19952623149688808_4_2024 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8364841 --exp_id DelayedCLS_Adv_4_0.19952623149688808_4_2013 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9824145 --exp_id DelayedCLS_Adv_4_0.25118864315095824_4_2017 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6857516 --exp_id DelayedCLS_Adv_4_0.25118864315095824_4_2020 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4929720 --exp_id DelayedCLS_Adv_4_0.25118864315095824_4_2022 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9988106 --exp_id DelayedCLS_Adv_4_0.25118864315095824_4_2024 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8307709 --exp_id DelayedCLS_Adv_4_0.25118864315095824_4_2013 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9118648 --exp_id DelayedCLS_Adv_4_0.31622776601683794_4_2017 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6478643 --exp_id DelayedCLS_Adv_4_0.31622776601683794_4_2020 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4066083 --exp_id DelayedCLS_Adv_4_0.31622776601683794_4_2022 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2735828 --exp_id DelayedCLS_Adv_4_0.31622776601683794_4_2024 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5863478 --exp_id DelayedCLS_Adv_4_0.31622776601683794_4_2013 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6469049 --exp_id DelayedCLS_Adv_4_0.3981071705534973_4_2017 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5732626 --exp_id DelayedCLS_Adv_4_0.3981071705534973_4_2020 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9938479 --exp_id DelayedCLS_Adv_4_0.3981071705534973_4_2022 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9935442 --exp_id DelayedCLS_Adv_4_0.3981071705534973_4_2024 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4888262 --exp_id DelayedCLS_Adv_4_0.3981071705534973_4_2013 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8276120 --exp_id DelayedCLS_Adv_4_0.5011872336272725_4_2017 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4278848 --exp_id DelayedCLS_Adv_4_0.5011872336272725_4_2020 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5792119 --exp_id DelayedCLS_Adv_4_0.5011872336272725_4_2022 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5777471 --exp_id DelayedCLS_Adv_4_0.5011872336272725_4_2024 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1789944 --exp_id DelayedCLS_Adv_4_0.5011872336272725_4_2013 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3433137 --exp_id DelayedCLS_Adv_4_0.6309573444801936_4_2017 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9112400 --exp_id DelayedCLS_Adv_4_0.6309573444801936_4_2020 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6222912 --exp_id DelayedCLS_Adv_4_0.6309573444801936_4_2022 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2514861 --exp_id DelayedCLS_Adv_4_0.6309573444801936_4_2024 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5273473 --exp_id DelayedCLS_Adv_4_0.6309573444801936_4_2013 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8011152 --exp_id DelayedCLS_Adv_4_0.7943282347242822_4_2017 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1444179 --exp_id DelayedCLS_Adv_4_0.7943282347242822_4_2020 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3388637 --exp_id DelayedCLS_Adv_4_0.7943282347242822_4_2022 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 549014 --exp_id DelayedCLS_Adv_4_0.7943282347242822_4_2024 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7880840 --exp_id DelayedCLS_Adv_4_0.7943282347242822_4_2013 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9320562 --exp_id DelayedCLS_Adv_4_1.0_4_2017 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 297846 --exp_id DelayedCLS_Adv_4_1.0_4_2020 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9858092 --exp_id DelayedCLS_Adv_4_1.0_4_2022 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8579014 --exp_id DelayedCLS_Adv_4_1.0_4_2024 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1482436 --exp_id DelayedCLS_Adv_4_1.0_4_2013 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 22665 --exp_id DelayedCLS_Adv_4_1.2589254117941675_4_2017 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6633406 --exp_id DelayedCLS_Adv_4_1.2589254117941675_4_2020 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 355737 --exp_id DelayedCLS_Adv_4_1.2589254117941675_4_2022 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1289091 --exp_id DelayedCLS_Adv_4_1.2589254117941675_4_2024 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8646404 --exp_id DelayedCLS_Adv_4_1.2589254117941675_4_2013 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1027118 --exp_id DelayedCLS_Adv_4_2.5118864315095824_4_2017 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3727749 --exp_id DelayedCLS_Adv_4_2.5118864315095824_4_2020 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6755300 --exp_id DelayedCLS_Adv_4_2.5118864315095824_4_2022 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1455029 --exp_id DelayedCLS_Adv_4_2.5118864315095824_4_2024 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5120721 --exp_id DelayedCLS_Adv_4_2.5118864315095824_4_2013 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9740599 --exp_id DelayedCLS_Adv_4_3.1622776601683795_4_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7491232 --exp_id DelayedCLS_Adv_4_3.1622776601683795_4_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8644386 --exp_id DelayedCLS_Adv_4_3.1622776601683795_4_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1326404 --exp_id DelayedCLS_Adv_4_3.1622776601683795_4_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2813909 --exp_id DelayedCLS_Adv_4_3.1622776601683795_4_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 529086 --exp_id DelayedCLS_Adv_4_5.011872336272725_4_2017 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3194064 --exp_id DelayedCLS_Adv_4_5.011872336272725_4_2020 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 411323 --exp_id DelayedCLS_Adv_4_5.011872336272725_4_2022 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1274054 --exp_id DelayedCLS_Adv_4_5.011872336272725_4_2024 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4544419 --exp_id DelayedCLS_Adv_4_5.011872336272725_4_2013 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6006556 --exp_id DelayedCLS_Adv_4_6.309573444801936_4_2017 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9646156 --exp_id DelayedCLS_Adv_4_6.309573444801936_4_2020 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3471519 --exp_id DelayedCLS_Adv_4_6.309573444801936_4_2022 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 165901 --exp_id DelayedCLS_Adv_4_6.309573444801936_4_2024 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9176480 --exp_id DelayedCLS_Adv_4_6.309573444801936_4_2013 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8061950 --exp_id DelayedCLS_Adv_4_7.943282347242821_4_2017 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6289633 --exp_id DelayedCLS_Adv_4_7.943282347242821_4_2020 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1707634 --exp_id DelayedCLS_Adv_4_7.943282347242821_4_2022 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8317628 --exp_id DelayedCLS_Adv_4_7.943282347242821_4_2024 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 405248 --exp_id DelayedCLS_Adv_4_7.943282347242821_4_2013 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 110807 --exp_id DelayedCLS_Adv_4_0.003981071705534973_4_2017 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2723679 --exp_id DelayedCLS_Adv_4_0.003981071705534973_4_2020 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3047434 --exp_id DelayedCLS_Adv_4_0.003981071705534973_4_2022 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1856543 --exp_id DelayedCLS_Adv_4_0.003981071705534973_4_2024 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7176645 --exp_id DelayedCLS_Adv_4_0.003981071705534973_4_2013 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2389992 --exp_id DelayedCLS_Adv_4_0.025118864315095808_4_2017 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 214689 --exp_id DelayedCLS_Adv_4_0.025118864315095808_4_2020 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8251889 --exp_id DelayedCLS_Adv_4_0.025118864315095808_4_2022 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6499538 --exp_id DelayedCLS_Adv_4_0.025118864315095808_4_2024 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4676810 --exp_id DelayedCLS_Adv_4_0.025118864315095808_4_2013 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6799015 --exp_id DelayedCLS_Adv_4_10.0_4_2017 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1041894 --exp_id DelayedCLS_Adv_4_10.0_4_2020 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7946614 --exp_id DelayedCLS_Adv_4_10.0_4_2022 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5369488 --exp_id DelayedCLS_Adv_4_10.0_4_2024 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6894555 --exp_id DelayedCLS_Adv_4_10.0_4_2013 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6482460 --exp_id DelayedCLS_Adv_4_12.589254117941687_4_2017 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2281770 --exp_id DelayedCLS_Adv_4_12.589254117941687_4_2020 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7384270 --exp_id DelayedCLS_Adv_4_12.589254117941687_4_2022 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9065957 --exp_id DelayedCLS_Adv_4_12.589254117941687_4_2024 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3593446 --exp_id DelayedCLS_Adv_4_12.589254117941687_4_2013 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3545171 --exp_id DelayedCLS_Adv_4_15.848931924611142_4_2017 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2202666 --exp_id DelayedCLS_Adv_4_15.848931924611142_4_2020 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9307289 --exp_id DelayedCLS_Adv_4_15.848931924611142_4_2022 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4507660 --exp_id DelayedCLS_Adv_4_15.848931924611142_4_2024 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7085416 --exp_id DelayedCLS_Adv_4_15.848931924611142_4_2013 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9690325 --exp_id DelayedCLS_Adv_4_19.952623149688787_4_2017 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5444424 --exp_id DelayedCLS_Adv_4_19.952623149688787_4_2020 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7764137 --exp_id DelayedCLS_Adv_4_19.952623149688787_4_2022 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5267016 --exp_id DelayedCLS_Adv_4_19.952623149688787_4_2024 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1834430 --exp_id DelayedCLS_Adv_4_19.952623149688787_4_2013 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3868296 --exp_id DelayedCLS_Adv_4_398.1071705534977_4_2017 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 299816 --exp_id DelayedCLS_Adv_4_398.1071705534977_4_2020 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 782011 --exp_id DelayedCLS_Adv_4_398.1071705534977_4_2022 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7160829 --exp_id DelayedCLS_Adv_4_398.1071705534977_4_2024 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2218387 --exp_id DelayedCLS_Adv_4_398.1071705534977_4_2013 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1585821 --exp_id DelayedCLS_Adv_4_0.0025118864315095794_4_2017 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6318320 --exp_id DelayedCLS_Adv_4_0.0025118864315095794_4_2020 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3438446 --exp_id DelayedCLS_Adv_4_0.0025118864315095794_4_2022 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 525850 --exp_id DelayedCLS_Adv_4_0.0025118864315095794_4_2024 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3791985 --exp_id DelayedCLS_Adv_4_0.0025118864315095794_4_2013 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5739772 --exp_id DelayedCLS_Adv_4_25.11886431509582_4_2017 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5172750 --exp_id DelayedCLS_Adv_4_25.11886431509582_4_2020 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5931939 --exp_id DelayedCLS_Adv_4_25.11886431509582_4_2022 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1753902 --exp_id DelayedCLS_Adv_4_25.11886431509582_4_2024 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9738494 --exp_id DelayedCLS_Adv_4_25.11886431509582_4_2013 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6366984 --exp_id DelayedCLS_Adv_4_794.3282347242822_4_2017 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7901026 --exp_id DelayedCLS_Adv_4_794.3282347242822_4_2020 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3744241 --exp_id DelayedCLS_Adv_4_794.3282347242822_4_2022 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4662865 --exp_id DelayedCLS_Adv_4_794.3282347242822_4_2024 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8441621 --exp_id DelayedCLS_Adv_4_794.3282347242822_4_2013 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9767236 --exp_id DelayedCLS_Adv_4_158.48931924611142_4_2017 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1652620 --exp_id DelayedCLS_Adv_4_158.48931924611142_4_2020 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2251410 --exp_id DelayedCLS_Adv_4_158.48931924611142_4_2022 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2758511 --exp_id DelayedCLS_Adv_4_158.48931924611142_4_2024 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1654636 --exp_id DelayedCLS_Adv_4_158.48931924611142_4_2013 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8936984 --exp_id DelayedCLS_Adv_4_31.622776601683793_4_2017 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8937558 --exp_id DelayedCLS_Adv_4_31.622776601683793_4_2020 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2485115 --exp_id DelayedCLS_Adv_4_31.622776601683793_4_2022 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1068447 --exp_id DelayedCLS_Adv_4_31.622776601683793_4_2024 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8766550 --exp_id DelayedCLS_Adv_4_31.622776601683793_4_2013 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5357497 --exp_id DelayedCLS_Adv_4_0.03162277660168379_4_2017 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5917568 --exp_id DelayedCLS_Adv_4_0.03162277660168379_4_2020 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9632435 --exp_id DelayedCLS_Adv_4_0.03162277660168379_4_2022 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 485185 --exp_id DelayedCLS_Adv_4_0.03162277660168379_4_2024 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 558448 --exp_id DelayedCLS_Adv_4_0.03162277660168379_4_2013 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8723419 --exp_id DelayedCLS_Adv_4_0.039810717055349734_4_2017 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4323621 --exp_id DelayedCLS_Adv_4_0.039810717055349734_4_2020 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2530934 --exp_id DelayedCLS_Adv_4_0.039810717055349734_4_2022 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1199687 --exp_id DelayedCLS_Adv_4_0.039810717055349734_4_2024 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 304510 --exp_id DelayedCLS_Adv_4_0.039810717055349734_4_2013 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7814814 --exp_id DelayedCLS_Adv_4_39.81071705534978_4_2017 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9826331 --exp_id DelayedCLS_Adv_4_39.81071705534978_4_2020 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1680898 --exp_id DelayedCLS_Adv_4_39.81071705534978_4_2022 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4192148 --exp_id DelayedCLS_Adv_4_39.81071705534978_4_2024 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2748870 --exp_id DelayedCLS_Adv_4_39.81071705534978_4_2013 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 375412 --exp_id DelayedCLS_Adv_4_0.005011872336272725_4_2017 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8733627 --exp_id DelayedCLS_Adv_4_0.005011872336272725_4_2020 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3267134 --exp_id DelayedCLS_Adv_4_0.005011872336272725_4_2022 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9831431 --exp_id DelayedCLS_Adv_4_0.005011872336272725_4_2024 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8905906 --exp_id DelayedCLS_Adv_4_0.005011872336272725_4_2013 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9566922 --exp_id DelayedCLS_Adv_4_0.01584893192461114_4_2017 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7811018 --exp_id DelayedCLS_Adv_4_0.01584893192461114_4_2020 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8009385 --exp_id DelayedCLS_Adv_4_0.01584893192461114_4_2022 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6545815 --exp_id DelayedCLS_Adv_4_0.01584893192461114_4_2024 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6316807 --exp_id DelayedCLS_Adv_4_0.01584893192461114_4_2013 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8818447 --exp_id DelayedCLS_Adv_4_3.981071705534973_4_2017 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3508369 --exp_id DelayedCLS_Adv_4_3.981071705534973_4_2020 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1760424 --exp_id DelayedCLS_Adv_4_3.981071705534973_4_2022 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 567372 --exp_id DelayedCLS_Adv_4_3.981071705534973_4_2024 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3644269 --exp_id DelayedCLS_Adv_4_3.981071705534973_4_2013 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9695446 --exp_id DelayedCLS_Adv_4_50.11872336272725_4_2017 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6874391 --exp_id DelayedCLS_Adv_4_50.11872336272725_4_2020 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 964738 --exp_id DelayedCLS_Adv_4_50.11872336272725_4_2022 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1616275 --exp_id DelayedCLS_Adv_4_50.11872336272725_4_2024 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 693620 --exp_id DelayedCLS_Adv_4_50.11872336272725_4_2013 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2175345 --exp_id DelayedCLS_Adv_4_0.0031622776601683794_4_2017 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1381645 --exp_id DelayedCLS_Adv_4_0.0031622776601683794_4_2020 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8383838 --exp_id DelayedCLS_Adv_4_0.0031622776601683794_4_2022 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1124809 --exp_id DelayedCLS_Adv_4_0.0031622776601683794_4_2024 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6855543 --exp_id DelayedCLS_Adv_4_0.0031622776601683794_4_2013 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7411475 --exp_id DelayedCLS_Adv_4_0.012589254117941675_4_2017 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1382686 --exp_id DelayedCLS_Adv_4_0.012589254117941675_4_2020 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1187512 --exp_id DelayedCLS_Adv_4_0.012589254117941675_4_2022 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8372751 --exp_id DelayedCLS_Adv_4_0.012589254117941675_4_2024 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 841016 --exp_id DelayedCLS_Adv_4_0.012589254117941675_4_2013 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7736480 --exp_id DelayedCLS_Adv_4_316.22776601683796_4_2017 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1377874 --exp_id DelayedCLS_Adv_4_316.22776601683796_4_2020 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8799851 --exp_id DelayedCLS_Adv_4_316.22776601683796_4_2022 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4474127 --exp_id DelayedCLS_Adv_4_316.22776601683796_4_2024 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3385166 --exp_id DelayedCLS_Adv_4_316.22776601683796_4_2013 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2312849 --exp_id DelayedCLS_Adv_4_63.09573444801943_4_2017 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5809626 --exp_id DelayedCLS_Adv_4_63.09573444801943_4_2020 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2102136 --exp_id DelayedCLS_Adv_4_63.09573444801943_4_2022 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9297783 --exp_id DelayedCLS_Adv_4_63.09573444801943_4_2024 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4218717 --exp_id DelayedCLS_Adv_4_63.09573444801943_4_2013 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3699473 --exp_id DelayedCLS_Adv_4_0.007943282347242814_4_2017 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1870875 --exp_id DelayedCLS_Adv_4_0.007943282347242814_4_2020 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5867277 --exp_id DelayedCLS_Adv_4_0.007943282347242814_4_2022 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3674834 --exp_id DelayedCLS_Adv_4_0.007943282347242814_4_2024 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9502294 --exp_id DelayedCLS_Adv_4_0.007943282347242814_4_2013 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 214276 --exp_id DelayedCLS_Adv_4_0.06309573444801933_4_2017 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8255776 --exp_id DelayedCLS_Adv_4_0.06309573444801933_4_2020 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1748938 --exp_id DelayedCLS_Adv_4_0.06309573444801933_4_2022 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6146581 --exp_id DelayedCLS_Adv_4_0.06309573444801933_4_2024 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7873627 --exp_id DelayedCLS_Adv_4_0.06309573444801933_4_2013 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8929823 --exp_id DelayedCLS_Adv_4_0.1_4_2017 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9348622 --exp_id DelayedCLS_Adv_4_0.1_4_2020 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6874537 --exp_id DelayedCLS_Adv_4_0.1_4_2022 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4101396 --exp_id DelayedCLS_Adv_4_0.1_4_2024 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3891878 --exp_id DelayedCLS_Adv_4_0.1_4_2013 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7430952 --exp_id DelayedCLS_Adv_4_0.12589254117941676_4_2017 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6571049 --exp_id DelayedCLS_Adv_4_0.12589254117941676_4_2020 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8073205 --exp_id DelayedCLS_Adv_4_0.12589254117941676_4_2022 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2219716 --exp_id DelayedCLS_Adv_4_0.12589254117941676_4_2024 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1810355 --exp_id DelayedCLS_Adv_4_0.12589254117941676_4_2013 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8868721 --exp_id DelayedCLS_Adv_4_199.5262314968883_4_2017 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5988357 --exp_id DelayedCLS_Adv_4_199.5262314968883_4_2020 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2477977 --exp_id DelayedCLS_Adv_4_199.5262314968883_4_2022 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6273231 --exp_id DelayedCLS_Adv_4_199.5262314968883_4_2024 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6258828 --exp_id DelayedCLS_Adv_4_199.5262314968883_4_2013 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8392669 --exp_id DelayedCLS_Adv_4_79.43282347242821_4_2017 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4914205 --exp_id DelayedCLS_Adv_4_79.43282347242821_4_2020 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2128342 --exp_id DelayedCLS_Adv_4_79.43282347242821_4_2022 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5607993 --exp_id DelayedCLS_Adv_4_79.43282347242821_4_2024 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7925609 --exp_id DelayedCLS_Adv_4_79.43282347242821_4_2013 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2239507 --exp_id DelayedCLS_Adv_4_0.0199526231496888_4_2017 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6409913 --exp_id DelayedCLS_Adv_4_0.0199526231496888_4_2020 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4766059 --exp_id DelayedCLS_Adv_4_0.0199526231496888_4_2022 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5918893 --exp_id DelayedCLS_Adv_4_0.0199526231496888_4_2024 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1236618 --exp_id DelayedCLS_Adv_4_0.0199526231496888_4_2013 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3609615 --exp_id DelayedCLS_Adv_4_0.001584893192461114_4_2017 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2169092 --exp_id DelayedCLS_Adv_4_0.001584893192461114_4_2020 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9983244 --exp_id DelayedCLS_Adv_4_0.001584893192461114_4_2022 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6287508 --exp_id DelayedCLS_Adv_4_0.001584893192461114_4_2024 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5684621 --exp_id DelayedCLS_Adv_4_0.001584893192461114_4_2013 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1547696 --exp_id DelayedCLS_Adv_4_0.05011872336272725_4_2017 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2427151 --exp_id DelayedCLS_Adv_4_0.05011872336272725_4_2020 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7789503 --exp_id DelayedCLS_Adv_4_0.05011872336272725_4_2022 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5430548 --exp_id DelayedCLS_Adv_4_0.05011872336272725_4_2024 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6837343 --exp_id DelayedCLS_Adv_4_0.05011872336272725_4_2013 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8187989 --exp_id DelayedCLS_Adv_4_1.584893192461114_4_2017 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8900583 --exp_id DelayedCLS_Adv_4_1.584893192461114_4_2020 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 613330 --exp_id DelayedCLS_Adv_4_1.584893192461114_4_2022 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1304869 --exp_id DelayedCLS_Adv_4_1.584893192461114_4_2024 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3132409 --exp_id DelayedCLS_Adv_4_1.584893192461114_4_2013 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4027452 --exp_id DelayedCLS_Adv_4_1.9952623149688808_4_2017 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7631135 --exp_id DelayedCLS_Adv_4_1.9952623149688808_4_2020 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5029148 --exp_id DelayedCLS_Adv_4_1.9952623149688808_4_2022 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8678616 --exp_id DelayedCLS_Adv_4_1.9952623149688808_4_2024 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3928315 --exp_id DelayedCLS_Adv_4_1.9952623149688808_4_2013 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7355895 --exp_id DelayedCLS_Adv_4_0.07943282347242818_4_2017 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4660835 --exp_id DelayedCLS_Adv_4_0.07943282347242818_4_2020 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 447619 --exp_id DelayedCLS_Adv_4_0.07943282347242818_4_2022 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2848838 --exp_id DelayedCLS_Adv_4_0.07943282347242818_4_2024 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6953809 --exp_id DelayedCLS_Adv_4_0.07943282347242818_4_2013 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9229481 --exp_id DelayedCLS_Adv_4_100.0_4_2017 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6169401 --exp_id DelayedCLS_Adv_4_100.0_4_2020 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4435277 --exp_id DelayedCLS_Adv_4_100.0_4_2022 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 403782 --exp_id DelayedCLS_Adv_4_100.0_4_2024 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4767693 --exp_id DelayedCLS_Adv_4_100.0_4_2013 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8086074 --exp_id DelayedCLS_Adv_4_1000.0_4_2017 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7390508 --exp_id DelayedCLS_Adv_4_1000.0_4_2020 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 640255 --exp_id DelayedCLS_Adv_4_1000.0_4_2022 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4580448 --exp_id DelayedCLS_Adv_4_1000.0_4_2024 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6201609 --exp_id DelayedCLS_Adv_4_1000.0_4_2013 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5243756 --exp_id DelayedCLS_Adv_4_0.0012589254117941675_4_2017 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7725848 --exp_id DelayedCLS_Adv_4_0.0012589254117941675_4_2020 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5513719 --exp_id DelayedCLS_Adv_4_0.0012589254117941675_4_2022 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1915073 --exp_id DelayedCLS_Adv_4_0.0012589254117941675_4_2024 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9128428 --exp_id DelayedCLS_Adv_4_0.0012589254117941675_4_2013 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1367750 --exp_id DelayedCLS_Adv_4_0.01_4_2017 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022247 --exp_id DelayedCLS_Adv_4_0.01_4_2020 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4511690 --exp_id DelayedCLS_Adv_4_0.01_4_2022 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 78433 --exp_id DelayedCLS_Adv_4_0.01_4_2024 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 468878 --exp_id DelayedCLS_Adv_4_0.01_4_2013 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7289826 --exp_id DelayedCLS_Adv_4_501.18723362727246_4_2017 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 577943 --exp_id DelayedCLS_Adv_4_501.18723362727246_4_2020 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5917896 --exp_id DelayedCLS_Adv_4_501.18723362727246_4_2022 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6993388 --exp_id DelayedCLS_Adv_4_501.18723362727246_4_2024 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1259437 --exp_id DelayedCLS_Adv_4_501.18723362727246_4_2013 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5994109 --exp_id DelayedCLS_Adv_4_630.9573444801943_4_2017 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6269841 --exp_id DelayedCLS_Adv_4_630.9573444801943_4_2020 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1604355 --exp_id DelayedCLS_Adv_4_630.9573444801943_4_2022 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6630930 --exp_id DelayedCLS_Adv_4_630.9573444801943_4_2024 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2269494 --exp_id DelayedCLS_Adv_4_630.9573444801943_4_2013 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2565157 --exp_id DelayedCLS_Adv_4_0.001995262314968879_4_2017 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8292842 --exp_id DelayedCLS_Adv_4_0.001995262314968879_4_2020 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5201597 --exp_id DelayedCLS_Adv_4_0.001995262314968879_4_2022 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1383049 --exp_id DelayedCLS_Adv_4_0.001995262314968879_4_2024 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4457488 --exp_id DelayedCLS_Adv_4_0.001995262314968879_4_2013 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7875652 --exp_id DelayedCLS_Adv_4_251.18864315095823_4_2017 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3225214 --exp_id DelayedCLS_Adv_4_251.18864315095823_4_2020 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7338820 --exp_id DelayedCLS_Adv_4_251.18864315095823_4_2022 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 808236 --exp_id DelayedCLS_Adv_4_251.18864315095823_4_2024 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6431526 --exp_id DelayedCLS_Adv_4_251.18864315095823_4_2013 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7145527 --exp_id DelayedCLS_Adv_4_125.89254117941688_4_2017 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3213018 --exp_id DelayedCLS_Adv_4_125.89254117941688_4_2020 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 806800 --exp_id DelayedCLS_Adv_4_125.89254117941688_4_2022 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2539949 --exp_id DelayedCLS_Adv_4_125.89254117941688_4_2024 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3087145 --exp_id DelayedCLS_Adv_4_125.89254117941688_4_2013 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6092505 --exp_id DelayedCLS_Adv_4_0.001_4_2017 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6560537 --exp_id DelayedCLS_Adv_4_0.001_4_2020 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9768080 --exp_id DelayedCLS_Adv_4_0.001_4_2022 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5772862 --exp_id DelayedCLS_Adv_4_0.001_4_2024 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2376980 --exp_id DelayedCLS_Adv_4_0.001_4_2013 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 4
sleep 2
