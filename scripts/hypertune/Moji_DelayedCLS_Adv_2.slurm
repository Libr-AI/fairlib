#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name=DelayedCLS_Adv_2

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_2_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_2_%J.out

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

python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6809913 --exp_id DelayedCLS_Adv_2_0.00630957344480193_2_2017 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4401707 --exp_id DelayedCLS_Adv_2_0.00630957344480193_2_2020 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4373986 --exp_id DelayedCLS_Adv_2_0.00630957344480193_2_2022 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2391485 --exp_id DelayedCLS_Adv_2_0.00630957344480193_2_2024 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8983771 --exp_id DelayedCLS_Adv_2_0.00630957344480193_2_2013 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6454813 --exp_id DelayedCLS_Adv_2_0.15848931924611143_2_2017 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6744180 --exp_id DelayedCLS_Adv_2_0.15848931924611143_2_2020 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 731806 --exp_id DelayedCLS_Adv_2_0.15848931924611143_2_2022 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3684262 --exp_id DelayedCLS_Adv_2_0.15848931924611143_2_2024 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1468846 --exp_id DelayedCLS_Adv_2_0.15848931924611143_2_2013 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2224282 --exp_id DelayedCLS_Adv_2_0.19952623149688808_2_2017 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1025824 --exp_id DelayedCLS_Adv_2_0.19952623149688808_2_2020 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7797578 --exp_id DelayedCLS_Adv_2_0.19952623149688808_2_2022 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7639668 --exp_id DelayedCLS_Adv_2_0.19952623149688808_2_2024 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3456185 --exp_id DelayedCLS_Adv_2_0.19952623149688808_2_2013 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3232984 --exp_id DelayedCLS_Adv_2_0.25118864315095824_2_2017 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3067895 --exp_id DelayedCLS_Adv_2_0.25118864315095824_2_2020 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1760266 --exp_id DelayedCLS_Adv_2_0.25118864315095824_2_2022 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5116244 --exp_id DelayedCLS_Adv_2_0.25118864315095824_2_2024 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3940343 --exp_id DelayedCLS_Adv_2_0.25118864315095824_2_2013 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4069470 --exp_id DelayedCLS_Adv_2_0.31622776601683794_2_2017 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6869402 --exp_id DelayedCLS_Adv_2_0.31622776601683794_2_2020 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 275748 --exp_id DelayedCLS_Adv_2_0.31622776601683794_2_2022 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 195261 --exp_id DelayedCLS_Adv_2_0.31622776601683794_2_2024 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7943249 --exp_id DelayedCLS_Adv_2_0.31622776601683794_2_2013 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3605513 --exp_id DelayedCLS_Adv_2_0.3981071705534973_2_2017 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 111342 --exp_id DelayedCLS_Adv_2_0.3981071705534973_2_2020 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9167839 --exp_id DelayedCLS_Adv_2_0.3981071705534973_2_2022 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9505912 --exp_id DelayedCLS_Adv_2_0.3981071705534973_2_2024 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7794976 --exp_id DelayedCLS_Adv_2_0.3981071705534973_2_2013 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4128199 --exp_id DelayedCLS_Adv_2_0.5011872336272725_2_2017 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6090930 --exp_id DelayedCLS_Adv_2_0.5011872336272725_2_2020 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1430726 --exp_id DelayedCLS_Adv_2_0.5011872336272725_2_2022 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1854312 --exp_id DelayedCLS_Adv_2_0.5011872336272725_2_2024 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3948290 --exp_id DelayedCLS_Adv_2_0.5011872336272725_2_2013 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7163821 --exp_id DelayedCLS_Adv_2_0.6309573444801936_2_2017 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 972980 --exp_id DelayedCLS_Adv_2_0.6309573444801936_2_2020 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2444353 --exp_id DelayedCLS_Adv_2_0.6309573444801936_2_2022 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3161209 --exp_id DelayedCLS_Adv_2_0.6309573444801936_2_2024 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2502431 --exp_id DelayedCLS_Adv_2_0.6309573444801936_2_2013 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8802259 --exp_id DelayedCLS_Adv_2_0.7943282347242822_2_2017 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6977429 --exp_id DelayedCLS_Adv_2_0.7943282347242822_2_2020 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1603000 --exp_id DelayedCLS_Adv_2_0.7943282347242822_2_2022 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2053345 --exp_id DelayedCLS_Adv_2_0.7943282347242822_2_2024 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3798615 --exp_id DelayedCLS_Adv_2_0.7943282347242822_2_2013 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 95544 --exp_id DelayedCLS_Adv_2_1.0_2_2017 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2330963 --exp_id DelayedCLS_Adv_2_1.0_2_2020 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5864181 --exp_id DelayedCLS_Adv_2_1.0_2_2022 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7414905 --exp_id DelayedCLS_Adv_2_1.0_2_2024 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3842206 --exp_id DelayedCLS_Adv_2_1.0_2_2013 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2168227 --exp_id DelayedCLS_Adv_2_1.2589254117941675_2_2017 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1569120 --exp_id DelayedCLS_Adv_2_1.2589254117941675_2_2020 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9928071 --exp_id DelayedCLS_Adv_2_1.2589254117941675_2_2022 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5810851 --exp_id DelayedCLS_Adv_2_1.2589254117941675_2_2024 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 25458 --exp_id DelayedCLS_Adv_2_1.2589254117941675_2_2013 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4148500 --exp_id DelayedCLS_Adv_2_2.5118864315095824_2_2017 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9998543 --exp_id DelayedCLS_Adv_2_2.5118864315095824_2_2020 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9346687 --exp_id DelayedCLS_Adv_2_2.5118864315095824_2_2022 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5068795 --exp_id DelayedCLS_Adv_2_2.5118864315095824_2_2024 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 708888 --exp_id DelayedCLS_Adv_2_2.5118864315095824_2_2013 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8726778 --exp_id DelayedCLS_Adv_2_3.1622776601683795_2_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3094074 --exp_id DelayedCLS_Adv_2_3.1622776601683795_2_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2041617 --exp_id DelayedCLS_Adv_2_3.1622776601683795_2_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2208573 --exp_id DelayedCLS_Adv_2_3.1622776601683795_2_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8376608 --exp_id DelayedCLS_Adv_2_3.1622776601683795_2_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8903057 --exp_id DelayedCLS_Adv_2_5.011872336272725_2_2017 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1372036 --exp_id DelayedCLS_Adv_2_5.011872336272725_2_2020 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3335977 --exp_id DelayedCLS_Adv_2_5.011872336272725_2_2022 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6856636 --exp_id DelayedCLS_Adv_2_5.011872336272725_2_2024 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2667970 --exp_id DelayedCLS_Adv_2_5.011872336272725_2_2013 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9034094 --exp_id DelayedCLS_Adv_2_6.309573444801936_2_2017 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2428883 --exp_id DelayedCLS_Adv_2_6.309573444801936_2_2020 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9193340 --exp_id DelayedCLS_Adv_2_6.309573444801936_2_2022 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7872653 --exp_id DelayedCLS_Adv_2_6.309573444801936_2_2024 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9758175 --exp_id DelayedCLS_Adv_2_6.309573444801936_2_2013 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5800977 --exp_id DelayedCLS_Adv_2_7.943282347242821_2_2017 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 959472 --exp_id DelayedCLS_Adv_2_7.943282347242821_2_2020 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4372142 --exp_id DelayedCLS_Adv_2_7.943282347242821_2_2022 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6141560 --exp_id DelayedCLS_Adv_2_7.943282347242821_2_2024 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7642522 --exp_id DelayedCLS_Adv_2_7.943282347242821_2_2013 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1796027 --exp_id DelayedCLS_Adv_2_0.003981071705534973_2_2017 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 657622 --exp_id DelayedCLS_Adv_2_0.003981071705534973_2_2020 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9584820 --exp_id DelayedCLS_Adv_2_0.003981071705534973_2_2022 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3735654 --exp_id DelayedCLS_Adv_2_0.003981071705534973_2_2024 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8986877 --exp_id DelayedCLS_Adv_2_0.003981071705534973_2_2013 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1337337 --exp_id DelayedCLS_Adv_2_0.025118864315095808_2_2017 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3453289 --exp_id DelayedCLS_Adv_2_0.025118864315095808_2_2020 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6392767 --exp_id DelayedCLS_Adv_2_0.025118864315095808_2_2022 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7720688 --exp_id DelayedCLS_Adv_2_0.025118864315095808_2_2024 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5872139 --exp_id DelayedCLS_Adv_2_0.025118864315095808_2_2013 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2867618 --exp_id DelayedCLS_Adv_2_10.0_2_2017 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5975738 --exp_id DelayedCLS_Adv_2_10.0_2_2020 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2864182 --exp_id DelayedCLS_Adv_2_10.0_2_2022 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2957831 --exp_id DelayedCLS_Adv_2_10.0_2_2024 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6873814 --exp_id DelayedCLS_Adv_2_10.0_2_2013 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1621201 --exp_id DelayedCLS_Adv_2_12.589254117941687_2_2017 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9096671 --exp_id DelayedCLS_Adv_2_12.589254117941687_2_2020 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3606856 --exp_id DelayedCLS_Adv_2_12.589254117941687_2_2022 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7558123 --exp_id DelayedCLS_Adv_2_12.589254117941687_2_2024 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5990706 --exp_id DelayedCLS_Adv_2_12.589254117941687_2_2013 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3785692 --exp_id DelayedCLS_Adv_2_15.848931924611142_2_2017 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5637674 --exp_id DelayedCLS_Adv_2_15.848931924611142_2_2020 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9128117 --exp_id DelayedCLS_Adv_2_15.848931924611142_2_2022 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9560278 --exp_id DelayedCLS_Adv_2_15.848931924611142_2_2024 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 878556 --exp_id DelayedCLS_Adv_2_15.848931924611142_2_2013 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4840163 --exp_id DelayedCLS_Adv_2_19.952623149688787_2_2017 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8371503 --exp_id DelayedCLS_Adv_2_19.952623149688787_2_2020 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9719735 --exp_id DelayedCLS_Adv_2_19.952623149688787_2_2022 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1753549 --exp_id DelayedCLS_Adv_2_19.952623149688787_2_2024 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2960367 --exp_id DelayedCLS_Adv_2_19.952623149688787_2_2013 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3221875 --exp_id DelayedCLS_Adv_2_398.1071705534977_2_2017 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8772430 --exp_id DelayedCLS_Adv_2_398.1071705534977_2_2020 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3950895 --exp_id DelayedCLS_Adv_2_398.1071705534977_2_2022 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9522602 --exp_id DelayedCLS_Adv_2_398.1071705534977_2_2024 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7431361 --exp_id DelayedCLS_Adv_2_398.1071705534977_2_2013 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 893088 --exp_id DelayedCLS_Adv_2_0.0025118864315095794_2_2017 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7985360 --exp_id DelayedCLS_Adv_2_0.0025118864315095794_2_2020 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8851260 --exp_id DelayedCLS_Adv_2_0.0025118864315095794_2_2022 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9941384 --exp_id DelayedCLS_Adv_2_0.0025118864315095794_2_2024 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 672995 --exp_id DelayedCLS_Adv_2_0.0025118864315095794_2_2013 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 523984 --exp_id DelayedCLS_Adv_2_25.11886431509582_2_2017 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6033391 --exp_id DelayedCLS_Adv_2_25.11886431509582_2_2020 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8189050 --exp_id DelayedCLS_Adv_2_25.11886431509582_2_2022 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8193689 --exp_id DelayedCLS_Adv_2_25.11886431509582_2_2024 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 804871 --exp_id DelayedCLS_Adv_2_25.11886431509582_2_2013 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1367722 --exp_id DelayedCLS_Adv_2_794.3282347242822_2_2017 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6715663 --exp_id DelayedCLS_Adv_2_794.3282347242822_2_2020 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8094525 --exp_id DelayedCLS_Adv_2_794.3282347242822_2_2022 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 620143 --exp_id DelayedCLS_Adv_2_794.3282347242822_2_2024 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 150947 --exp_id DelayedCLS_Adv_2_794.3282347242822_2_2013 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4921867 --exp_id DelayedCLS_Adv_2_158.48931924611142_2_2017 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1773327 --exp_id DelayedCLS_Adv_2_158.48931924611142_2_2020 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 824707 --exp_id DelayedCLS_Adv_2_158.48931924611142_2_2022 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 842142 --exp_id DelayedCLS_Adv_2_158.48931924611142_2_2024 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6363831 --exp_id DelayedCLS_Adv_2_158.48931924611142_2_2013 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7650192 --exp_id DelayedCLS_Adv_2_31.622776601683793_2_2017 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 347839 --exp_id DelayedCLS_Adv_2_31.622776601683793_2_2020 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5465348 --exp_id DelayedCLS_Adv_2_31.622776601683793_2_2022 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1782596 --exp_id DelayedCLS_Adv_2_31.622776601683793_2_2024 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1299162 --exp_id DelayedCLS_Adv_2_31.622776601683793_2_2013 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2909055 --exp_id DelayedCLS_Adv_2_0.03162277660168379_2_2017 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2067989 --exp_id DelayedCLS_Adv_2_0.03162277660168379_2_2020 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4808765 --exp_id DelayedCLS_Adv_2_0.03162277660168379_2_2022 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8927288 --exp_id DelayedCLS_Adv_2_0.03162277660168379_2_2024 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4082510 --exp_id DelayedCLS_Adv_2_0.03162277660168379_2_2013 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8642010 --exp_id DelayedCLS_Adv_2_0.039810717055349734_2_2017 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9972817 --exp_id DelayedCLS_Adv_2_0.039810717055349734_2_2020 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9483405 --exp_id DelayedCLS_Adv_2_0.039810717055349734_2_2022 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2152718 --exp_id DelayedCLS_Adv_2_0.039810717055349734_2_2024 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3063318 --exp_id DelayedCLS_Adv_2_0.039810717055349734_2_2013 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8048236 --exp_id DelayedCLS_Adv_2_39.81071705534978_2_2017 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3264241 --exp_id DelayedCLS_Adv_2_39.81071705534978_2_2020 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1455292 --exp_id DelayedCLS_Adv_2_39.81071705534978_2_2022 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 742392 --exp_id DelayedCLS_Adv_2_39.81071705534978_2_2024 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3251240 --exp_id DelayedCLS_Adv_2_39.81071705534978_2_2013 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2646478 --exp_id DelayedCLS_Adv_2_0.005011872336272725_2_2017 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 269498 --exp_id DelayedCLS_Adv_2_0.005011872336272725_2_2020 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1790722 --exp_id DelayedCLS_Adv_2_0.005011872336272725_2_2022 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8487101 --exp_id DelayedCLS_Adv_2_0.005011872336272725_2_2024 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4948714 --exp_id DelayedCLS_Adv_2_0.005011872336272725_2_2013 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4998171 --exp_id DelayedCLS_Adv_2_0.01584893192461114_2_2017 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8769909 --exp_id DelayedCLS_Adv_2_0.01584893192461114_2_2020 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9691245 --exp_id DelayedCLS_Adv_2_0.01584893192461114_2_2022 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9256725 --exp_id DelayedCLS_Adv_2_0.01584893192461114_2_2024 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3780096 --exp_id DelayedCLS_Adv_2_0.01584893192461114_2_2013 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9660557 --exp_id DelayedCLS_Adv_2_3.981071705534973_2_2017 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3314827 --exp_id DelayedCLS_Adv_2_3.981071705534973_2_2020 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8216672 --exp_id DelayedCLS_Adv_2_3.981071705534973_2_2022 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2315781 --exp_id DelayedCLS_Adv_2_3.981071705534973_2_2024 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9588677 --exp_id DelayedCLS_Adv_2_3.981071705534973_2_2013 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8945596 --exp_id DelayedCLS_Adv_2_50.11872336272725_2_2017 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 440659 --exp_id DelayedCLS_Adv_2_50.11872336272725_2_2020 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9475472 --exp_id DelayedCLS_Adv_2_50.11872336272725_2_2022 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1294286 --exp_id DelayedCLS_Adv_2_50.11872336272725_2_2024 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8796671 --exp_id DelayedCLS_Adv_2_50.11872336272725_2_2013 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 967913 --exp_id DelayedCLS_Adv_2_0.0031622776601683794_2_2017 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4678891 --exp_id DelayedCLS_Adv_2_0.0031622776601683794_2_2020 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6681631 --exp_id DelayedCLS_Adv_2_0.0031622776601683794_2_2022 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 745953 --exp_id DelayedCLS_Adv_2_0.0031622776601683794_2_2024 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7427041 --exp_id DelayedCLS_Adv_2_0.0031622776601683794_2_2013 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9578739 --exp_id DelayedCLS_Adv_2_0.012589254117941675_2_2017 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3698951 --exp_id DelayedCLS_Adv_2_0.012589254117941675_2_2020 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9078712 --exp_id DelayedCLS_Adv_2_0.012589254117941675_2_2022 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5955234 --exp_id DelayedCLS_Adv_2_0.012589254117941675_2_2024 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4397609 --exp_id DelayedCLS_Adv_2_0.012589254117941675_2_2013 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4406632 --exp_id DelayedCLS_Adv_2_316.22776601683796_2_2017 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2457217 --exp_id DelayedCLS_Adv_2_316.22776601683796_2_2020 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 606503 --exp_id DelayedCLS_Adv_2_316.22776601683796_2_2022 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8689947 --exp_id DelayedCLS_Adv_2_316.22776601683796_2_2024 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2496862 --exp_id DelayedCLS_Adv_2_316.22776601683796_2_2013 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4362862 --exp_id DelayedCLS_Adv_2_63.09573444801943_2_2017 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5393410 --exp_id DelayedCLS_Adv_2_63.09573444801943_2_2020 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9455952 --exp_id DelayedCLS_Adv_2_63.09573444801943_2_2022 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9177523 --exp_id DelayedCLS_Adv_2_63.09573444801943_2_2024 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1537072 --exp_id DelayedCLS_Adv_2_63.09573444801943_2_2013 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1302446 --exp_id DelayedCLS_Adv_2_0.007943282347242814_2_2017 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9859833 --exp_id DelayedCLS_Adv_2_0.007943282347242814_2_2020 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7508762 --exp_id DelayedCLS_Adv_2_0.007943282347242814_2_2022 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2710426 --exp_id DelayedCLS_Adv_2_0.007943282347242814_2_2024 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9412190 --exp_id DelayedCLS_Adv_2_0.007943282347242814_2_2013 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7687092 --exp_id DelayedCLS_Adv_2_0.06309573444801933_2_2017 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6293176 --exp_id DelayedCLS_Adv_2_0.06309573444801933_2_2020 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1097033 --exp_id DelayedCLS_Adv_2_0.06309573444801933_2_2022 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4341811 --exp_id DelayedCLS_Adv_2_0.06309573444801933_2_2024 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 804893 --exp_id DelayedCLS_Adv_2_0.06309573444801933_2_2013 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8397012 --exp_id DelayedCLS_Adv_2_0.1_2_2017 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5022986 --exp_id DelayedCLS_Adv_2_0.1_2_2020 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1390357 --exp_id DelayedCLS_Adv_2_0.1_2_2022 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9413924 --exp_id DelayedCLS_Adv_2_0.1_2_2024 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9860010 --exp_id DelayedCLS_Adv_2_0.1_2_2013 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5399250 --exp_id DelayedCLS_Adv_2_0.12589254117941676_2_2017 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7222000 --exp_id DelayedCLS_Adv_2_0.12589254117941676_2_2020 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6543203 --exp_id DelayedCLS_Adv_2_0.12589254117941676_2_2022 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7431600 --exp_id DelayedCLS_Adv_2_0.12589254117941676_2_2024 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3250420 --exp_id DelayedCLS_Adv_2_0.12589254117941676_2_2013 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4922141 --exp_id DelayedCLS_Adv_2_199.5262314968883_2_2017 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5241019 --exp_id DelayedCLS_Adv_2_199.5262314968883_2_2020 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1866892 --exp_id DelayedCLS_Adv_2_199.5262314968883_2_2022 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2492496 --exp_id DelayedCLS_Adv_2_199.5262314968883_2_2024 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5438307 --exp_id DelayedCLS_Adv_2_199.5262314968883_2_2013 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5635404 --exp_id DelayedCLS_Adv_2_79.43282347242821_2_2017 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9822209 --exp_id DelayedCLS_Adv_2_79.43282347242821_2_2020 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9655083 --exp_id DelayedCLS_Adv_2_79.43282347242821_2_2022 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6784560 --exp_id DelayedCLS_Adv_2_79.43282347242821_2_2024 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9396902 --exp_id DelayedCLS_Adv_2_79.43282347242821_2_2013 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7897734 --exp_id DelayedCLS_Adv_2_0.0199526231496888_2_2017 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2275837 --exp_id DelayedCLS_Adv_2_0.0199526231496888_2_2020 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9111316 --exp_id DelayedCLS_Adv_2_0.0199526231496888_2_2022 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 693224 --exp_id DelayedCLS_Adv_2_0.0199526231496888_2_2024 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4620740 --exp_id DelayedCLS_Adv_2_0.0199526231496888_2_2013 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4557924 --exp_id DelayedCLS_Adv_2_0.001584893192461114_2_2017 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4272310 --exp_id DelayedCLS_Adv_2_0.001584893192461114_2_2020 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9836506 --exp_id DelayedCLS_Adv_2_0.001584893192461114_2_2022 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5859859 --exp_id DelayedCLS_Adv_2_0.001584893192461114_2_2024 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 73284 --exp_id DelayedCLS_Adv_2_0.001584893192461114_2_2013 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7231569 --exp_id DelayedCLS_Adv_2_0.05011872336272725_2_2017 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8498385 --exp_id DelayedCLS_Adv_2_0.05011872336272725_2_2020 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2225060 --exp_id DelayedCLS_Adv_2_0.05011872336272725_2_2022 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4939271 --exp_id DelayedCLS_Adv_2_0.05011872336272725_2_2024 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3544100 --exp_id DelayedCLS_Adv_2_0.05011872336272725_2_2013 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9424605 --exp_id DelayedCLS_Adv_2_1.584893192461114_2_2017 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3791364 --exp_id DelayedCLS_Adv_2_1.584893192461114_2_2020 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6372522 --exp_id DelayedCLS_Adv_2_1.584893192461114_2_2022 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3843278 --exp_id DelayedCLS_Adv_2_1.584893192461114_2_2024 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1820835 --exp_id DelayedCLS_Adv_2_1.584893192461114_2_2013 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3022882 --exp_id DelayedCLS_Adv_2_1.9952623149688808_2_2017 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7101216 --exp_id DelayedCLS_Adv_2_1.9952623149688808_2_2020 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7256494 --exp_id DelayedCLS_Adv_2_1.9952623149688808_2_2022 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6218499 --exp_id DelayedCLS_Adv_2_1.9952623149688808_2_2024 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6027637 --exp_id DelayedCLS_Adv_2_1.9952623149688808_2_2013 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3520587 --exp_id DelayedCLS_Adv_2_0.07943282347242818_2_2017 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 466851 --exp_id DelayedCLS_Adv_2_0.07943282347242818_2_2020 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5625188 --exp_id DelayedCLS_Adv_2_0.07943282347242818_2_2022 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 330242 --exp_id DelayedCLS_Adv_2_0.07943282347242818_2_2024 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5546679 --exp_id DelayedCLS_Adv_2_0.07943282347242818_2_2013 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6186419 --exp_id DelayedCLS_Adv_2_100.0_2_2017 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8826223 --exp_id DelayedCLS_Adv_2_100.0_2_2020 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1109605 --exp_id DelayedCLS_Adv_2_100.0_2_2022 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2603134 --exp_id DelayedCLS_Adv_2_100.0_2_2024 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3453944 --exp_id DelayedCLS_Adv_2_100.0_2_2013 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9007418 --exp_id DelayedCLS_Adv_2_1000.0_2_2017 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9813226 --exp_id DelayedCLS_Adv_2_1000.0_2_2020 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5603097 --exp_id DelayedCLS_Adv_2_1000.0_2_2022 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6512550 --exp_id DelayedCLS_Adv_2_1000.0_2_2024 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8437170 --exp_id DelayedCLS_Adv_2_1000.0_2_2013 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 972629 --exp_id DelayedCLS_Adv_2_0.0012589254117941675_2_2017 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5456272 --exp_id DelayedCLS_Adv_2_0.0012589254117941675_2_2020 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8841749 --exp_id DelayedCLS_Adv_2_0.0012589254117941675_2_2022 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8344342 --exp_id DelayedCLS_Adv_2_0.0012589254117941675_2_2024 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6574399 --exp_id DelayedCLS_Adv_2_0.0012589254117941675_2_2013 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 858909 --exp_id DelayedCLS_Adv_2_0.01_2_2017 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4581997 --exp_id DelayedCLS_Adv_2_0.01_2_2020 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5178630 --exp_id DelayedCLS_Adv_2_0.01_2_2022 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1171003 --exp_id DelayedCLS_Adv_2_0.01_2_2024 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4667315 --exp_id DelayedCLS_Adv_2_0.01_2_2013 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5204225 --exp_id DelayedCLS_Adv_2_501.18723362727246_2_2017 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6043989 --exp_id DelayedCLS_Adv_2_501.18723362727246_2_2020 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7611496 --exp_id DelayedCLS_Adv_2_501.18723362727246_2_2022 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7149058 --exp_id DelayedCLS_Adv_2_501.18723362727246_2_2024 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2517855 --exp_id DelayedCLS_Adv_2_501.18723362727246_2_2013 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4677106 --exp_id DelayedCLS_Adv_2_630.9573444801943_2_2017 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1021387 --exp_id DelayedCLS_Adv_2_630.9573444801943_2_2020 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7998541 --exp_id DelayedCLS_Adv_2_630.9573444801943_2_2022 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 999829 --exp_id DelayedCLS_Adv_2_630.9573444801943_2_2024 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7577465 --exp_id DelayedCLS_Adv_2_630.9573444801943_2_2013 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2676536 --exp_id DelayedCLS_Adv_2_0.001995262314968879_2_2017 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 738724 --exp_id DelayedCLS_Adv_2_0.001995262314968879_2_2020 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9657611 --exp_id DelayedCLS_Adv_2_0.001995262314968879_2_2022 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1752072 --exp_id DelayedCLS_Adv_2_0.001995262314968879_2_2024 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6310242 --exp_id DelayedCLS_Adv_2_0.001995262314968879_2_2013 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7268388 --exp_id DelayedCLS_Adv_2_251.18864315095823_2_2017 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3540572 --exp_id DelayedCLS_Adv_2_251.18864315095823_2_2020 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7760058 --exp_id DelayedCLS_Adv_2_251.18864315095823_2_2022 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1088930 --exp_id DelayedCLS_Adv_2_251.18864315095823_2_2024 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7010444 --exp_id DelayedCLS_Adv_2_251.18864315095823_2_2013 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6945464 --exp_id DelayedCLS_Adv_2_125.89254117941688_2_2017 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9963090 --exp_id DelayedCLS_Adv_2_125.89254117941688_2_2020 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5909263 --exp_id DelayedCLS_Adv_2_125.89254117941688_2_2022 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8874898 --exp_id DelayedCLS_Adv_2_125.89254117941688_2_2024 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9304239 --exp_id DelayedCLS_Adv_2_125.89254117941688_2_2013 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1016952 --exp_id DelayedCLS_Adv_2_0.001_2_2017 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8358551 --exp_id DelayedCLS_Adv_2_0.001_2_2020 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5163758 --exp_id DelayedCLS_Adv_2_0.001_2_2022 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1075858 --exp_id DelayedCLS_Adv_2_0.001_2_2024 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7025511 --exp_id DelayedCLS_Adv_2_0.001_2_2013 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 2
sleep 2
