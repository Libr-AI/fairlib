#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name=DelayedCLS_Adv_10

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_10_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_10_%J.out

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

python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1349014 --exp_id DelayedCLS_Adv_10_0.00630957344480193_10_2017 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3991618 --exp_id DelayedCLS_Adv_10_0.00630957344480193_10_2020 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7904882 --exp_id DelayedCLS_Adv_10_0.00630957344480193_10_2022 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4365480 --exp_id DelayedCLS_Adv_10_0.00630957344480193_10_2024 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8667329 --exp_id DelayedCLS_Adv_10_0.00630957344480193_10_2013 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7760574 --exp_id DelayedCLS_Adv_10_0.15848931924611143_10_2017 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 418010 --exp_id DelayedCLS_Adv_10_0.15848931924611143_10_2020 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8238044 --exp_id DelayedCLS_Adv_10_0.15848931924611143_10_2022 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7214212 --exp_id DelayedCLS_Adv_10_0.15848931924611143_10_2024 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7991109 --exp_id DelayedCLS_Adv_10_0.15848931924611143_10_2013 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2164334 --exp_id DelayedCLS_Adv_10_0.19952623149688808_10_2017 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8311436 --exp_id DelayedCLS_Adv_10_0.19952623149688808_10_2020 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8527189 --exp_id DelayedCLS_Adv_10_0.19952623149688808_10_2022 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8274162 --exp_id DelayedCLS_Adv_10_0.19952623149688808_10_2024 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8865526 --exp_id DelayedCLS_Adv_10_0.19952623149688808_10_2013 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 941180 --exp_id DelayedCLS_Adv_10_0.25118864315095824_10_2017 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1529517 --exp_id DelayedCLS_Adv_10_0.25118864315095824_10_2020 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6712276 --exp_id DelayedCLS_Adv_10_0.25118864315095824_10_2022 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7294391 --exp_id DelayedCLS_Adv_10_0.25118864315095824_10_2024 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1135433 --exp_id DelayedCLS_Adv_10_0.25118864315095824_10_2013 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5198133 --exp_id DelayedCLS_Adv_10_0.31622776601683794_10_2017 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 110263 --exp_id DelayedCLS_Adv_10_0.31622776601683794_10_2020 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1654382 --exp_id DelayedCLS_Adv_10_0.31622776601683794_10_2022 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7543411 --exp_id DelayedCLS_Adv_10_0.31622776601683794_10_2024 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 710975 --exp_id DelayedCLS_Adv_10_0.31622776601683794_10_2013 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 55109 --exp_id DelayedCLS_Adv_10_0.3981071705534973_10_2017 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9032112 --exp_id DelayedCLS_Adv_10_0.3981071705534973_10_2020 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4357244 --exp_id DelayedCLS_Adv_10_0.3981071705534973_10_2022 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1132118 --exp_id DelayedCLS_Adv_10_0.3981071705534973_10_2024 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2274949 --exp_id DelayedCLS_Adv_10_0.3981071705534973_10_2013 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2185541 --exp_id DelayedCLS_Adv_10_0.5011872336272725_10_2017 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8463102 --exp_id DelayedCLS_Adv_10_0.5011872336272725_10_2020 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5831274 --exp_id DelayedCLS_Adv_10_0.5011872336272725_10_2022 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6530413 --exp_id DelayedCLS_Adv_10_0.5011872336272725_10_2024 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9998479 --exp_id DelayedCLS_Adv_10_0.5011872336272725_10_2013 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3457416 --exp_id DelayedCLS_Adv_10_0.6309573444801936_10_2017 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4130752 --exp_id DelayedCLS_Adv_10_0.6309573444801936_10_2020 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3171969 --exp_id DelayedCLS_Adv_10_0.6309573444801936_10_2022 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3088961 --exp_id DelayedCLS_Adv_10_0.6309573444801936_10_2024 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8492559 --exp_id DelayedCLS_Adv_10_0.6309573444801936_10_2013 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2922767 --exp_id DelayedCLS_Adv_10_0.7943282347242822_10_2017 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6311389 --exp_id DelayedCLS_Adv_10_0.7943282347242822_10_2020 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8655916 --exp_id DelayedCLS_Adv_10_0.7943282347242822_10_2022 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7686831 --exp_id DelayedCLS_Adv_10_0.7943282347242822_10_2024 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4511848 --exp_id DelayedCLS_Adv_10_0.7943282347242822_10_2013 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1993959 --exp_id DelayedCLS_Adv_10_1.0_10_2017 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9072722 --exp_id DelayedCLS_Adv_10_1.0_10_2020 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1582068 --exp_id DelayedCLS_Adv_10_1.0_10_2022 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 432974 --exp_id DelayedCLS_Adv_10_1.0_10_2024 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5008750 --exp_id DelayedCLS_Adv_10_1.0_10_2013 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5251392 --exp_id DelayedCLS_Adv_10_1.2589254117941675_10_2017 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3189298 --exp_id DelayedCLS_Adv_10_1.2589254117941675_10_2020 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9739852 --exp_id DelayedCLS_Adv_10_1.2589254117941675_10_2022 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1172168 --exp_id DelayedCLS_Adv_10_1.2589254117941675_10_2024 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6365982 --exp_id DelayedCLS_Adv_10_1.2589254117941675_10_2013 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5332689 --exp_id DelayedCLS_Adv_10_2.5118864315095824_10_2017 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3638062 --exp_id DelayedCLS_Adv_10_2.5118864315095824_10_2020 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5397303 --exp_id DelayedCLS_Adv_10_2.5118864315095824_10_2022 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 461636 --exp_id DelayedCLS_Adv_10_2.5118864315095824_10_2024 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9553451 --exp_id DelayedCLS_Adv_10_2.5118864315095824_10_2013 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6352525 --exp_id DelayedCLS_Adv_10_3.1622776601683795_10_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4537627 --exp_id DelayedCLS_Adv_10_3.1622776601683795_10_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5627680 --exp_id DelayedCLS_Adv_10_3.1622776601683795_10_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3671291 --exp_id DelayedCLS_Adv_10_3.1622776601683795_10_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3324236 --exp_id DelayedCLS_Adv_10_3.1622776601683795_10_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1302726 --exp_id DelayedCLS_Adv_10_5.011872336272725_10_2017 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8640126 --exp_id DelayedCLS_Adv_10_5.011872336272725_10_2020 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4882662 --exp_id DelayedCLS_Adv_10_5.011872336272725_10_2022 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3442428 --exp_id DelayedCLS_Adv_10_5.011872336272725_10_2024 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5149412 --exp_id DelayedCLS_Adv_10_5.011872336272725_10_2013 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2497667 --exp_id DelayedCLS_Adv_10_6.309573444801936_10_2017 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2316797 --exp_id DelayedCLS_Adv_10_6.309573444801936_10_2020 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5528826 --exp_id DelayedCLS_Adv_10_6.309573444801936_10_2022 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6157651 --exp_id DelayedCLS_Adv_10_6.309573444801936_10_2024 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6750365 --exp_id DelayedCLS_Adv_10_6.309573444801936_10_2013 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6409736 --exp_id DelayedCLS_Adv_10_7.943282347242821_10_2017 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7646494 --exp_id DelayedCLS_Adv_10_7.943282347242821_10_2020 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1665812 --exp_id DelayedCLS_Adv_10_7.943282347242821_10_2022 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2348224 --exp_id DelayedCLS_Adv_10_7.943282347242821_10_2024 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8531744 --exp_id DelayedCLS_Adv_10_7.943282347242821_10_2013 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 53230 --exp_id DelayedCLS_Adv_10_0.003981071705534973_10_2017 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 888528 --exp_id DelayedCLS_Adv_10_0.003981071705534973_10_2020 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3868192 --exp_id DelayedCLS_Adv_10_0.003981071705534973_10_2022 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3641847 --exp_id DelayedCLS_Adv_10_0.003981071705534973_10_2024 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1478194 --exp_id DelayedCLS_Adv_10_0.003981071705534973_10_2013 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4925365 --exp_id DelayedCLS_Adv_10_0.025118864315095808_10_2017 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8271525 --exp_id DelayedCLS_Adv_10_0.025118864315095808_10_2020 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 788579 --exp_id DelayedCLS_Adv_10_0.025118864315095808_10_2022 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8698925 --exp_id DelayedCLS_Adv_10_0.025118864315095808_10_2024 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9818128 --exp_id DelayedCLS_Adv_10_0.025118864315095808_10_2013 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3556366 --exp_id DelayedCLS_Adv_10_10.0_10_2017 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3852817 --exp_id DelayedCLS_Adv_10_10.0_10_2020 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5063917 --exp_id DelayedCLS_Adv_10_10.0_10_2022 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 192892 --exp_id DelayedCLS_Adv_10_10.0_10_2024 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7982100 --exp_id DelayedCLS_Adv_10_10.0_10_2013 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2303142 --exp_id DelayedCLS_Adv_10_12.589254117941687_10_2017 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1608575 --exp_id DelayedCLS_Adv_10_12.589254117941687_10_2020 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3451897 --exp_id DelayedCLS_Adv_10_12.589254117941687_10_2022 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2574570 --exp_id DelayedCLS_Adv_10_12.589254117941687_10_2024 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1041148 --exp_id DelayedCLS_Adv_10_12.589254117941687_10_2013 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8799946 --exp_id DelayedCLS_Adv_10_15.848931924611142_10_2017 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3705588 --exp_id DelayedCLS_Adv_10_15.848931924611142_10_2020 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 988882 --exp_id DelayedCLS_Adv_10_15.848931924611142_10_2022 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9873075 --exp_id DelayedCLS_Adv_10_15.848931924611142_10_2024 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7933586 --exp_id DelayedCLS_Adv_10_15.848931924611142_10_2013 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5294627 --exp_id DelayedCLS_Adv_10_19.952623149688787_10_2017 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8626430 --exp_id DelayedCLS_Adv_10_19.952623149688787_10_2020 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3425764 --exp_id DelayedCLS_Adv_10_19.952623149688787_10_2022 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 537511 --exp_id DelayedCLS_Adv_10_19.952623149688787_10_2024 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8107513 --exp_id DelayedCLS_Adv_10_19.952623149688787_10_2013 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1475358 --exp_id DelayedCLS_Adv_10_398.1071705534977_10_2017 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5875977 --exp_id DelayedCLS_Adv_10_398.1071705534977_10_2020 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6865892 --exp_id DelayedCLS_Adv_10_398.1071705534977_10_2022 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3964403 --exp_id DelayedCLS_Adv_10_398.1071705534977_10_2024 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3545788 --exp_id DelayedCLS_Adv_10_398.1071705534977_10_2013 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8673477 --exp_id DelayedCLS_Adv_10_0.0025118864315095794_10_2017 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7969638 --exp_id DelayedCLS_Adv_10_0.0025118864315095794_10_2020 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9057949 --exp_id DelayedCLS_Adv_10_0.0025118864315095794_10_2022 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2814796 --exp_id DelayedCLS_Adv_10_0.0025118864315095794_10_2024 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1684925 --exp_id DelayedCLS_Adv_10_0.0025118864315095794_10_2013 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 609207 --exp_id DelayedCLS_Adv_10_25.11886431509582_10_2017 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6845571 --exp_id DelayedCLS_Adv_10_25.11886431509582_10_2020 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8568696 --exp_id DelayedCLS_Adv_10_25.11886431509582_10_2022 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3835810 --exp_id DelayedCLS_Adv_10_25.11886431509582_10_2024 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8303129 --exp_id DelayedCLS_Adv_10_25.11886431509582_10_2013 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6248712 --exp_id DelayedCLS_Adv_10_794.3282347242822_10_2017 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7548499 --exp_id DelayedCLS_Adv_10_794.3282347242822_10_2020 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4034905 --exp_id DelayedCLS_Adv_10_794.3282347242822_10_2022 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6150997 --exp_id DelayedCLS_Adv_10_794.3282347242822_10_2024 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2833104 --exp_id DelayedCLS_Adv_10_794.3282347242822_10_2013 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1074180 --exp_id DelayedCLS_Adv_10_158.48931924611142_10_2017 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6114677 --exp_id DelayedCLS_Adv_10_158.48931924611142_10_2020 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3845775 --exp_id DelayedCLS_Adv_10_158.48931924611142_10_2022 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5885086 --exp_id DelayedCLS_Adv_10_158.48931924611142_10_2024 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2053160 --exp_id DelayedCLS_Adv_10_158.48931924611142_10_2013 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6133193 --exp_id DelayedCLS_Adv_10_31.622776601683793_10_2017 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4433722 --exp_id DelayedCLS_Adv_10_31.622776601683793_10_2020 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7054977 --exp_id DelayedCLS_Adv_10_31.622776601683793_10_2022 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 987721 --exp_id DelayedCLS_Adv_10_31.622776601683793_10_2024 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2466094 --exp_id DelayedCLS_Adv_10_31.622776601683793_10_2013 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1178401 --exp_id DelayedCLS_Adv_10_0.03162277660168379_10_2017 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3270784 --exp_id DelayedCLS_Adv_10_0.03162277660168379_10_2020 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7314514 --exp_id DelayedCLS_Adv_10_0.03162277660168379_10_2022 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1390906 --exp_id DelayedCLS_Adv_10_0.03162277660168379_10_2024 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8868645 --exp_id DelayedCLS_Adv_10_0.03162277660168379_10_2013 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5166235 --exp_id DelayedCLS_Adv_10_0.039810717055349734_10_2017 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6747719 --exp_id DelayedCLS_Adv_10_0.039810717055349734_10_2020 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6590321 --exp_id DelayedCLS_Adv_10_0.039810717055349734_10_2022 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3475659 --exp_id DelayedCLS_Adv_10_0.039810717055349734_10_2024 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7249235 --exp_id DelayedCLS_Adv_10_0.039810717055349734_10_2013 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6817492 --exp_id DelayedCLS_Adv_10_39.81071705534978_10_2017 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8592689 --exp_id DelayedCLS_Adv_10_39.81071705534978_10_2020 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7891543 --exp_id DelayedCLS_Adv_10_39.81071705534978_10_2022 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5086683 --exp_id DelayedCLS_Adv_10_39.81071705534978_10_2024 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5843668 --exp_id DelayedCLS_Adv_10_39.81071705534978_10_2013 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7301725 --exp_id DelayedCLS_Adv_10_0.005011872336272725_10_2017 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 46625 --exp_id DelayedCLS_Adv_10_0.005011872336272725_10_2020 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1591726 --exp_id DelayedCLS_Adv_10_0.005011872336272725_10_2022 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 101437 --exp_id DelayedCLS_Adv_10_0.005011872336272725_10_2024 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8918827 --exp_id DelayedCLS_Adv_10_0.005011872336272725_10_2013 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9954312 --exp_id DelayedCLS_Adv_10_0.01584893192461114_10_2017 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4337681 --exp_id DelayedCLS_Adv_10_0.01584893192461114_10_2020 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7725578 --exp_id DelayedCLS_Adv_10_0.01584893192461114_10_2022 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6781019 --exp_id DelayedCLS_Adv_10_0.01584893192461114_10_2024 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2386464 --exp_id DelayedCLS_Adv_10_0.01584893192461114_10_2013 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9911427 --exp_id DelayedCLS_Adv_10_3.981071705534973_10_2017 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5143936 --exp_id DelayedCLS_Adv_10_3.981071705534973_10_2020 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7541358 --exp_id DelayedCLS_Adv_10_3.981071705534973_10_2022 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 427587 --exp_id DelayedCLS_Adv_10_3.981071705534973_10_2024 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 33948 --exp_id DelayedCLS_Adv_10_3.981071705534973_10_2013 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1346385 --exp_id DelayedCLS_Adv_10_50.11872336272725_10_2017 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 149667 --exp_id DelayedCLS_Adv_10_50.11872336272725_10_2020 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7602024 --exp_id DelayedCLS_Adv_10_50.11872336272725_10_2022 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6360846 --exp_id DelayedCLS_Adv_10_50.11872336272725_10_2024 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 907229 --exp_id DelayedCLS_Adv_10_50.11872336272725_10_2013 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4756794 --exp_id DelayedCLS_Adv_10_0.0031622776601683794_10_2017 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 331190 --exp_id DelayedCLS_Adv_10_0.0031622776601683794_10_2020 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9085350 --exp_id DelayedCLS_Adv_10_0.0031622776601683794_10_2022 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2989391 --exp_id DelayedCLS_Adv_10_0.0031622776601683794_10_2024 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 660198 --exp_id DelayedCLS_Adv_10_0.0031622776601683794_10_2013 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1145838 --exp_id DelayedCLS_Adv_10_0.012589254117941675_10_2017 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1215803 --exp_id DelayedCLS_Adv_10_0.012589254117941675_10_2020 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2483414 --exp_id DelayedCLS_Adv_10_0.012589254117941675_10_2022 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8491570 --exp_id DelayedCLS_Adv_10_0.012589254117941675_10_2024 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9642379 --exp_id DelayedCLS_Adv_10_0.012589254117941675_10_2013 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 895093 --exp_id DelayedCLS_Adv_10_316.22776601683796_10_2017 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5301683 --exp_id DelayedCLS_Adv_10_316.22776601683796_10_2020 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9699494 --exp_id DelayedCLS_Adv_10_316.22776601683796_10_2022 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4545230 --exp_id DelayedCLS_Adv_10_316.22776601683796_10_2024 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5532569 --exp_id DelayedCLS_Adv_10_316.22776601683796_10_2013 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1047767 --exp_id DelayedCLS_Adv_10_63.09573444801943_10_2017 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2221741 --exp_id DelayedCLS_Adv_10_63.09573444801943_10_2020 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8880008 --exp_id DelayedCLS_Adv_10_63.09573444801943_10_2022 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7079222 --exp_id DelayedCLS_Adv_10_63.09573444801943_10_2024 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4423541 --exp_id DelayedCLS_Adv_10_63.09573444801943_10_2013 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2212900 --exp_id DelayedCLS_Adv_10_0.007943282347242814_10_2017 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9567877 --exp_id DelayedCLS_Adv_10_0.007943282347242814_10_2020 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4966089 --exp_id DelayedCLS_Adv_10_0.007943282347242814_10_2022 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7322113 --exp_id DelayedCLS_Adv_10_0.007943282347242814_10_2024 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1414660 --exp_id DelayedCLS_Adv_10_0.007943282347242814_10_2013 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9541615 --exp_id DelayedCLS_Adv_10_0.06309573444801933_10_2017 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5008598 --exp_id DelayedCLS_Adv_10_0.06309573444801933_10_2020 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4195955 --exp_id DelayedCLS_Adv_10_0.06309573444801933_10_2022 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1067344 --exp_id DelayedCLS_Adv_10_0.06309573444801933_10_2024 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2462798 --exp_id DelayedCLS_Adv_10_0.06309573444801933_10_2013 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8808783 --exp_id DelayedCLS_Adv_10_0.1_10_2017 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5552273 --exp_id DelayedCLS_Adv_10_0.1_10_2020 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6293986 --exp_id DelayedCLS_Adv_10_0.1_10_2022 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4889156 --exp_id DelayedCLS_Adv_10_0.1_10_2024 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 231136 --exp_id DelayedCLS_Adv_10_0.1_10_2013 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5720098 --exp_id DelayedCLS_Adv_10_0.12589254117941676_10_2017 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7491728 --exp_id DelayedCLS_Adv_10_0.12589254117941676_10_2020 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8798803 --exp_id DelayedCLS_Adv_10_0.12589254117941676_10_2022 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9347550 --exp_id DelayedCLS_Adv_10_0.12589254117941676_10_2024 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2805656 --exp_id DelayedCLS_Adv_10_0.12589254117941676_10_2013 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6408267 --exp_id DelayedCLS_Adv_10_199.5262314968883_10_2017 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9193719 --exp_id DelayedCLS_Adv_10_199.5262314968883_10_2020 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8396334 --exp_id DelayedCLS_Adv_10_199.5262314968883_10_2022 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5488077 --exp_id DelayedCLS_Adv_10_199.5262314968883_10_2024 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4089931 --exp_id DelayedCLS_Adv_10_199.5262314968883_10_2013 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3472010 --exp_id DelayedCLS_Adv_10_79.43282347242821_10_2017 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9938611 --exp_id DelayedCLS_Adv_10_79.43282347242821_10_2020 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8538161 --exp_id DelayedCLS_Adv_10_79.43282347242821_10_2022 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 245046 --exp_id DelayedCLS_Adv_10_79.43282347242821_10_2024 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4907759 --exp_id DelayedCLS_Adv_10_79.43282347242821_10_2013 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3933748 --exp_id DelayedCLS_Adv_10_0.0199526231496888_10_2017 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1439154 --exp_id DelayedCLS_Adv_10_0.0199526231496888_10_2020 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8209610 --exp_id DelayedCLS_Adv_10_0.0199526231496888_10_2022 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 741729 --exp_id DelayedCLS_Adv_10_0.0199526231496888_10_2024 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9824961 --exp_id DelayedCLS_Adv_10_0.0199526231496888_10_2013 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4051116 --exp_id DelayedCLS_Adv_10_0.001584893192461114_10_2017 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8119881 --exp_id DelayedCLS_Adv_10_0.001584893192461114_10_2020 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5195812 --exp_id DelayedCLS_Adv_10_0.001584893192461114_10_2022 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7197290 --exp_id DelayedCLS_Adv_10_0.001584893192461114_10_2024 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5944172 --exp_id DelayedCLS_Adv_10_0.001584893192461114_10_2013 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1075139 --exp_id DelayedCLS_Adv_10_0.05011872336272725_10_2017 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3835292 --exp_id DelayedCLS_Adv_10_0.05011872336272725_10_2020 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3454729 --exp_id DelayedCLS_Adv_10_0.05011872336272725_10_2022 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 602006 --exp_id DelayedCLS_Adv_10_0.05011872336272725_10_2024 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 542092 --exp_id DelayedCLS_Adv_10_0.05011872336272725_10_2013 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2649913 --exp_id DelayedCLS_Adv_10_1.584893192461114_10_2017 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1676230 --exp_id DelayedCLS_Adv_10_1.584893192461114_10_2020 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5618040 --exp_id DelayedCLS_Adv_10_1.584893192461114_10_2022 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5824078 --exp_id DelayedCLS_Adv_10_1.584893192461114_10_2024 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3951568 --exp_id DelayedCLS_Adv_10_1.584893192461114_10_2013 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8884077 --exp_id DelayedCLS_Adv_10_1.9952623149688808_10_2017 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4676897 --exp_id DelayedCLS_Adv_10_1.9952623149688808_10_2020 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1658812 --exp_id DelayedCLS_Adv_10_1.9952623149688808_10_2022 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8790134 --exp_id DelayedCLS_Adv_10_1.9952623149688808_10_2024 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6301001 --exp_id DelayedCLS_Adv_10_1.9952623149688808_10_2013 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7608333 --exp_id DelayedCLS_Adv_10_0.07943282347242818_10_2017 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2120612 --exp_id DelayedCLS_Adv_10_0.07943282347242818_10_2020 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 34132 --exp_id DelayedCLS_Adv_10_0.07943282347242818_10_2022 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9982301 --exp_id DelayedCLS_Adv_10_0.07943282347242818_10_2024 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3122532 --exp_id DelayedCLS_Adv_10_0.07943282347242818_10_2013 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 105849 --exp_id DelayedCLS_Adv_10_100.0_10_2017 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1664486 --exp_id DelayedCLS_Adv_10_100.0_10_2020 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5855238 --exp_id DelayedCLS_Adv_10_100.0_10_2022 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 379860 --exp_id DelayedCLS_Adv_10_100.0_10_2024 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2971615 --exp_id DelayedCLS_Adv_10_100.0_10_2013 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6059997 --exp_id DelayedCLS_Adv_10_1000.0_10_2017 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1874922 --exp_id DelayedCLS_Adv_10_1000.0_10_2020 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3173155 --exp_id DelayedCLS_Adv_10_1000.0_10_2022 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5890795 --exp_id DelayedCLS_Adv_10_1000.0_10_2024 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 31154 --exp_id DelayedCLS_Adv_10_1000.0_10_2013 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6697060 --exp_id DelayedCLS_Adv_10_0.0012589254117941675_10_2017 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5423405 --exp_id DelayedCLS_Adv_10_0.0012589254117941675_10_2020 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6285272 --exp_id DelayedCLS_Adv_10_0.0012589254117941675_10_2022 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4507946 --exp_id DelayedCLS_Adv_10_0.0012589254117941675_10_2024 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 701677 --exp_id DelayedCLS_Adv_10_0.0012589254117941675_10_2013 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4724018 --exp_id DelayedCLS_Adv_10_0.01_10_2017 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9679190 --exp_id DelayedCLS_Adv_10_0.01_10_2020 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5886687 --exp_id DelayedCLS_Adv_10_0.01_10_2022 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9945230 --exp_id DelayedCLS_Adv_10_0.01_10_2024 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4074655 --exp_id DelayedCLS_Adv_10_0.01_10_2013 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4871736 --exp_id DelayedCLS_Adv_10_501.18723362727246_10_2017 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5194712 --exp_id DelayedCLS_Adv_10_501.18723362727246_10_2020 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7894893 --exp_id DelayedCLS_Adv_10_501.18723362727246_10_2022 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3935942 --exp_id DelayedCLS_Adv_10_501.18723362727246_10_2024 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2585000 --exp_id DelayedCLS_Adv_10_501.18723362727246_10_2013 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8526221 --exp_id DelayedCLS_Adv_10_630.9573444801943_10_2017 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7804122 --exp_id DelayedCLS_Adv_10_630.9573444801943_10_2020 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4382933 --exp_id DelayedCLS_Adv_10_630.9573444801943_10_2022 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5060154 --exp_id DelayedCLS_Adv_10_630.9573444801943_10_2024 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6197951 --exp_id DelayedCLS_Adv_10_630.9573444801943_10_2013 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4685357 --exp_id DelayedCLS_Adv_10_0.001995262314968879_10_2017 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2216265 --exp_id DelayedCLS_Adv_10_0.001995262314968879_10_2020 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5384171 --exp_id DelayedCLS_Adv_10_0.001995262314968879_10_2022 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5772261 --exp_id DelayedCLS_Adv_10_0.001995262314968879_10_2024 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6414174 --exp_id DelayedCLS_Adv_10_0.001995262314968879_10_2013 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 959574 --exp_id DelayedCLS_Adv_10_251.18864315095823_10_2017 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 874790 --exp_id DelayedCLS_Adv_10_251.18864315095823_10_2020 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4487431 --exp_id DelayedCLS_Adv_10_251.18864315095823_10_2022 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3201572 --exp_id DelayedCLS_Adv_10_251.18864315095823_10_2024 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7168469 --exp_id DelayedCLS_Adv_10_251.18864315095823_10_2013 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7445609 --exp_id DelayedCLS_Adv_10_125.89254117941688_10_2017 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5687214 --exp_id DelayedCLS_Adv_10_125.89254117941688_10_2020 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2115291 --exp_id DelayedCLS_Adv_10_125.89254117941688_10_2022 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6644427 --exp_id DelayedCLS_Adv_10_125.89254117941688_10_2024 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6298770 --exp_id DelayedCLS_Adv_10_125.89254117941688_10_2013 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1355470 --exp_id DelayedCLS_Adv_10_0.001_10_2017 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5117731 --exp_id DelayedCLS_Adv_10_0.001_10_2020 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5714722 --exp_id DelayedCLS_Adv_10_0.001_10_2022 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8908173 --exp_id DelayedCLS_Adv_10_0.001_10_2024 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3662404 --exp_id DelayedCLS_Adv_10_0.001_10_2013 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 10
sleep 2
