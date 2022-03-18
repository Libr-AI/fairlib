#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name=DelayedCLS_Adv_100

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_100_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/DelayedCLS_Adv_100_%J.out

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

python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5057210 --exp_id DelayedCLS_Adv_100_0.00630957344480193_100_2017 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8353183 --exp_id DelayedCLS_Adv_100_0.00630957344480193_100_2020 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1139504 --exp_id DelayedCLS_Adv_100_0.00630957344480193_100_2022 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4746641 --exp_id DelayedCLS_Adv_100_0.00630957344480193_100_2024 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7262987 --exp_id DelayedCLS_Adv_100_0.00630957344480193_100_2013 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2916792 --exp_id DelayedCLS_Adv_100_0.15848931924611143_100_2017 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4716922 --exp_id DelayedCLS_Adv_100_0.15848931924611143_100_2020 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5728757 --exp_id DelayedCLS_Adv_100_0.15848931924611143_100_2022 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8613369 --exp_id DelayedCLS_Adv_100_0.15848931924611143_100_2024 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6073270 --exp_id DelayedCLS_Adv_100_0.15848931924611143_100_2013 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3689773 --exp_id DelayedCLS_Adv_100_0.19952623149688808_100_2017 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3885733 --exp_id DelayedCLS_Adv_100_0.19952623149688808_100_2020 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2926123 --exp_id DelayedCLS_Adv_100_0.19952623149688808_100_2022 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2737900 --exp_id DelayedCLS_Adv_100_0.19952623149688808_100_2024 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7720948 --exp_id DelayedCLS_Adv_100_0.19952623149688808_100_2013 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6370639 --exp_id DelayedCLS_Adv_100_0.25118864315095824_100_2017 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 844677 --exp_id DelayedCLS_Adv_100_0.25118864315095824_100_2020 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3587735 --exp_id DelayedCLS_Adv_100_0.25118864315095824_100_2022 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9236423 --exp_id DelayedCLS_Adv_100_0.25118864315095824_100_2024 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4769036 --exp_id DelayedCLS_Adv_100_0.25118864315095824_100_2013 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2360601 --exp_id DelayedCLS_Adv_100_0.31622776601683794_100_2017 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3566394 --exp_id DelayedCLS_Adv_100_0.31622776601683794_100_2020 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6161501 --exp_id DelayedCLS_Adv_100_0.31622776601683794_100_2022 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8233275 --exp_id DelayedCLS_Adv_100_0.31622776601683794_100_2024 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 350121 --exp_id DelayedCLS_Adv_100_0.31622776601683794_100_2013 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6814352 --exp_id DelayedCLS_Adv_100_0.3981071705534973_100_2017 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 400543 --exp_id DelayedCLS_Adv_100_0.3981071705534973_100_2020 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7685289 --exp_id DelayedCLS_Adv_100_0.3981071705534973_100_2022 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6227079 --exp_id DelayedCLS_Adv_100_0.3981071705534973_100_2024 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3853068 --exp_id DelayedCLS_Adv_100_0.3981071705534973_100_2013 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4878857 --exp_id DelayedCLS_Adv_100_0.5011872336272725_100_2017 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 659359 --exp_id DelayedCLS_Adv_100_0.5011872336272725_100_2020 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4828320 --exp_id DelayedCLS_Adv_100_0.5011872336272725_100_2022 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6483029 --exp_id DelayedCLS_Adv_100_0.5011872336272725_100_2024 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2390790 --exp_id DelayedCLS_Adv_100_0.5011872336272725_100_2013 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4543159 --exp_id DelayedCLS_Adv_100_0.6309573444801936_100_2017 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6915208 --exp_id DelayedCLS_Adv_100_0.6309573444801936_100_2020 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7966088 --exp_id DelayedCLS_Adv_100_0.6309573444801936_100_2022 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6096383 --exp_id DelayedCLS_Adv_100_0.6309573444801936_100_2024 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6163636 --exp_id DelayedCLS_Adv_100_0.6309573444801936_100_2013 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1525253 --exp_id DelayedCLS_Adv_100_0.7943282347242822_100_2017 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9173957 --exp_id DelayedCLS_Adv_100_0.7943282347242822_100_2020 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4839081 --exp_id DelayedCLS_Adv_100_0.7943282347242822_100_2022 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9988950 --exp_id DelayedCLS_Adv_100_0.7943282347242822_100_2024 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3681135 --exp_id DelayedCLS_Adv_100_0.7943282347242822_100_2013 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3952111 --exp_id DelayedCLS_Adv_100_1.0_100_2017 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7806563 --exp_id DelayedCLS_Adv_100_1.0_100_2020 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2436363 --exp_id DelayedCLS_Adv_100_1.0_100_2022 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5431490 --exp_id DelayedCLS_Adv_100_1.0_100_2024 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 793795 --exp_id DelayedCLS_Adv_100_1.0_100_2013 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1253286 --exp_id DelayedCLS_Adv_100_1.2589254117941675_100_2017 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7603373 --exp_id DelayedCLS_Adv_100_1.2589254117941675_100_2020 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3463273 --exp_id DelayedCLS_Adv_100_1.2589254117941675_100_2022 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8564235 --exp_id DelayedCLS_Adv_100_1.2589254117941675_100_2024 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9774866 --exp_id DelayedCLS_Adv_100_1.2589254117941675_100_2013 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7995647 --exp_id DelayedCLS_Adv_100_2.5118864315095824_100_2017 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8071584 --exp_id DelayedCLS_Adv_100_2.5118864315095824_100_2020 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1853634 --exp_id DelayedCLS_Adv_100_2.5118864315095824_100_2022 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8051398 --exp_id DelayedCLS_Adv_100_2.5118864315095824_100_2024 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6435608 --exp_id DelayedCLS_Adv_100_2.5118864315095824_100_2013 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2402779 --exp_id DelayedCLS_Adv_100_3.1622776601683795_100_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5456180 --exp_id DelayedCLS_Adv_100_3.1622776601683795_100_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3357066 --exp_id DelayedCLS_Adv_100_3.1622776601683795_100_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9465724 --exp_id DelayedCLS_Adv_100_3.1622776601683795_100_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7014846 --exp_id DelayedCLS_Adv_100_3.1622776601683795_100_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2460797 --exp_id DelayedCLS_Adv_100_5.011872336272725_100_2017 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9442366 --exp_id DelayedCLS_Adv_100_5.011872336272725_100_2020 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1748636 --exp_id DelayedCLS_Adv_100_5.011872336272725_100_2022 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2666364 --exp_id DelayedCLS_Adv_100_5.011872336272725_100_2024 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 399325 --exp_id DelayedCLS_Adv_100_5.011872336272725_100_2013 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2456053 --exp_id DelayedCLS_Adv_100_6.309573444801936_100_2017 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2641396 --exp_id DelayedCLS_Adv_100_6.309573444801936_100_2020 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4892652 --exp_id DelayedCLS_Adv_100_6.309573444801936_100_2022 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 44152 --exp_id DelayedCLS_Adv_100_6.309573444801936_100_2024 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6247393 --exp_id DelayedCLS_Adv_100_6.309573444801936_100_2013 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3473895 --exp_id DelayedCLS_Adv_100_7.943282347242821_100_2017 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9690928 --exp_id DelayedCLS_Adv_100_7.943282347242821_100_2020 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2141940 --exp_id DelayedCLS_Adv_100_7.943282347242821_100_2022 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7555852 --exp_id DelayedCLS_Adv_100_7.943282347242821_100_2024 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5235432 --exp_id DelayedCLS_Adv_100_7.943282347242821_100_2013 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5154983 --exp_id DelayedCLS_Adv_100_0.003981071705534973_100_2017 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7909674 --exp_id DelayedCLS_Adv_100_0.003981071705534973_100_2020 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7658211 --exp_id DelayedCLS_Adv_100_0.003981071705534973_100_2022 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4717423 --exp_id DelayedCLS_Adv_100_0.003981071705534973_100_2024 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2164730 --exp_id DelayedCLS_Adv_100_0.003981071705534973_100_2013 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2361355 --exp_id DelayedCLS_Adv_100_0.025118864315095808_100_2017 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9864977 --exp_id DelayedCLS_Adv_100_0.025118864315095808_100_2020 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9116456 --exp_id DelayedCLS_Adv_100_0.025118864315095808_100_2022 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4805355 --exp_id DelayedCLS_Adv_100_0.025118864315095808_100_2024 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6811451 --exp_id DelayedCLS_Adv_100_0.025118864315095808_100_2013 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9556200 --exp_id DelayedCLS_Adv_100_10.0_100_2017 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4654303 --exp_id DelayedCLS_Adv_100_10.0_100_2020 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4482125 --exp_id DelayedCLS_Adv_100_10.0_100_2022 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4007829 --exp_id DelayedCLS_Adv_100_10.0_100_2024 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2797205 --exp_id DelayedCLS_Adv_100_10.0_100_2013 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2048607 --exp_id DelayedCLS_Adv_100_12.589254117941687_100_2017 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3472937 --exp_id DelayedCLS_Adv_100_12.589254117941687_100_2020 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6359066 --exp_id DelayedCLS_Adv_100_12.589254117941687_100_2022 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9380091 --exp_id DelayedCLS_Adv_100_12.589254117941687_100_2024 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7410149 --exp_id DelayedCLS_Adv_100_12.589254117941687_100_2013 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1297844 --exp_id DelayedCLS_Adv_100_15.848931924611142_100_2017 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3264627 --exp_id DelayedCLS_Adv_100_15.848931924611142_100_2020 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 263828 --exp_id DelayedCLS_Adv_100_15.848931924611142_100_2022 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7596268 --exp_id DelayedCLS_Adv_100_15.848931924611142_100_2024 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8681593 --exp_id DelayedCLS_Adv_100_15.848931924611142_100_2013 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3219202 --exp_id DelayedCLS_Adv_100_19.952623149688787_100_2017 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 175863 --exp_id DelayedCLS_Adv_100_19.952623149688787_100_2020 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4325407 --exp_id DelayedCLS_Adv_100_19.952623149688787_100_2022 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4677488 --exp_id DelayedCLS_Adv_100_19.952623149688787_100_2024 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4433273 --exp_id DelayedCLS_Adv_100_19.952623149688787_100_2013 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9803214 --exp_id DelayedCLS_Adv_100_398.1071705534977_100_2017 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7473863 --exp_id DelayedCLS_Adv_100_398.1071705534977_100_2020 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1305781 --exp_id DelayedCLS_Adv_100_398.1071705534977_100_2022 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3675933 --exp_id DelayedCLS_Adv_100_398.1071705534977_100_2024 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6518348 --exp_id DelayedCLS_Adv_100_398.1071705534977_100_2013 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1362068 --exp_id DelayedCLS_Adv_100_0.0025118864315095794_100_2017 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3568563 --exp_id DelayedCLS_Adv_100_0.0025118864315095794_100_2020 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7435797 --exp_id DelayedCLS_Adv_100_0.0025118864315095794_100_2022 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6610702 --exp_id DelayedCLS_Adv_100_0.0025118864315095794_100_2024 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9041113 --exp_id DelayedCLS_Adv_100_0.0025118864315095794_100_2013 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 822162 --exp_id DelayedCLS_Adv_100_25.11886431509582_100_2017 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7961744 --exp_id DelayedCLS_Adv_100_25.11886431509582_100_2020 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9524760 --exp_id DelayedCLS_Adv_100_25.11886431509582_100_2022 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3149637 --exp_id DelayedCLS_Adv_100_25.11886431509582_100_2024 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1884160 --exp_id DelayedCLS_Adv_100_25.11886431509582_100_2013 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6498156 --exp_id DelayedCLS_Adv_100_794.3282347242822_100_2017 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4804938 --exp_id DelayedCLS_Adv_100_794.3282347242822_100_2020 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5983788 --exp_id DelayedCLS_Adv_100_794.3282347242822_100_2022 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9252616 --exp_id DelayedCLS_Adv_100_794.3282347242822_100_2024 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8238965 --exp_id DelayedCLS_Adv_100_794.3282347242822_100_2013 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4975255 --exp_id DelayedCLS_Adv_100_158.48931924611142_100_2017 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4146249 --exp_id DelayedCLS_Adv_100_158.48931924611142_100_2020 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 83356 --exp_id DelayedCLS_Adv_100_158.48931924611142_100_2022 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1255249 --exp_id DelayedCLS_Adv_100_158.48931924611142_100_2024 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2110097 --exp_id DelayedCLS_Adv_100_158.48931924611142_100_2013 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9160629 --exp_id DelayedCLS_Adv_100_31.622776601683793_100_2017 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6201210 --exp_id DelayedCLS_Adv_100_31.622776601683793_100_2020 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5351080 --exp_id DelayedCLS_Adv_100_31.622776601683793_100_2022 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9833952 --exp_id DelayedCLS_Adv_100_31.622776601683793_100_2024 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1031349 --exp_id DelayedCLS_Adv_100_31.622776601683793_100_2013 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6860998 --exp_id DelayedCLS_Adv_100_0.03162277660168379_100_2017 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4372772 --exp_id DelayedCLS_Adv_100_0.03162277660168379_100_2020 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7752804 --exp_id DelayedCLS_Adv_100_0.03162277660168379_100_2022 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4270481 --exp_id DelayedCLS_Adv_100_0.03162277660168379_100_2024 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 383995 --exp_id DelayedCLS_Adv_100_0.03162277660168379_100_2013 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8469550 --exp_id DelayedCLS_Adv_100_0.039810717055349734_100_2017 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4219770 --exp_id DelayedCLS_Adv_100_0.039810717055349734_100_2020 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5783473 --exp_id DelayedCLS_Adv_100_0.039810717055349734_100_2022 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5775727 --exp_id DelayedCLS_Adv_100_0.039810717055349734_100_2024 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4064197 --exp_id DelayedCLS_Adv_100_0.039810717055349734_100_2013 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4081498 --exp_id DelayedCLS_Adv_100_39.81071705534978_100_2017 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6324783 --exp_id DelayedCLS_Adv_100_39.81071705534978_100_2020 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8334578 --exp_id DelayedCLS_Adv_100_39.81071705534978_100_2022 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3630283 --exp_id DelayedCLS_Adv_100_39.81071705534978_100_2024 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9871675 --exp_id DelayedCLS_Adv_100_39.81071705534978_100_2013 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8226149 --exp_id DelayedCLS_Adv_100_0.005011872336272725_100_2017 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5918583 --exp_id DelayedCLS_Adv_100_0.005011872336272725_100_2020 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1647341 --exp_id DelayedCLS_Adv_100_0.005011872336272725_100_2022 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1537648 --exp_id DelayedCLS_Adv_100_0.005011872336272725_100_2024 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 967349 --exp_id DelayedCLS_Adv_100_0.005011872336272725_100_2013 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7658141 --exp_id DelayedCLS_Adv_100_0.01584893192461114_100_2017 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2756438 --exp_id DelayedCLS_Adv_100_0.01584893192461114_100_2020 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6780827 --exp_id DelayedCLS_Adv_100_0.01584893192461114_100_2022 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8718034 --exp_id DelayedCLS_Adv_100_0.01584893192461114_100_2024 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8480223 --exp_id DelayedCLS_Adv_100_0.01584893192461114_100_2013 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 279329 --exp_id DelayedCLS_Adv_100_3.981071705534973_100_2017 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6655308 --exp_id DelayedCLS_Adv_100_3.981071705534973_100_2020 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8635047 --exp_id DelayedCLS_Adv_100_3.981071705534973_100_2022 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2390796 --exp_id DelayedCLS_Adv_100_3.981071705534973_100_2024 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6570752 --exp_id DelayedCLS_Adv_100_3.981071705534973_100_2013 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5555744 --exp_id DelayedCLS_Adv_100_50.11872336272725_100_2017 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2871819 --exp_id DelayedCLS_Adv_100_50.11872336272725_100_2020 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5639498 --exp_id DelayedCLS_Adv_100_50.11872336272725_100_2022 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8693494 --exp_id DelayedCLS_Adv_100_50.11872336272725_100_2024 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3277197 --exp_id DelayedCLS_Adv_100_50.11872336272725_100_2013 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3107608 --exp_id DelayedCLS_Adv_100_0.0031622776601683794_100_2017 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6442989 --exp_id DelayedCLS_Adv_100_0.0031622776601683794_100_2020 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9598504 --exp_id DelayedCLS_Adv_100_0.0031622776601683794_100_2022 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5919495 --exp_id DelayedCLS_Adv_100_0.0031622776601683794_100_2024 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3758410 --exp_id DelayedCLS_Adv_100_0.0031622776601683794_100_2013 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6370296 --exp_id DelayedCLS_Adv_100_0.012589254117941675_100_2017 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8799501 --exp_id DelayedCLS_Adv_100_0.012589254117941675_100_2020 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6202296 --exp_id DelayedCLS_Adv_100_0.012589254117941675_100_2022 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7035437 --exp_id DelayedCLS_Adv_100_0.012589254117941675_100_2024 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5535531 --exp_id DelayedCLS_Adv_100_0.012589254117941675_100_2013 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9285689 --exp_id DelayedCLS_Adv_100_316.22776601683796_100_2017 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9244884 --exp_id DelayedCLS_Adv_100_316.22776601683796_100_2020 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8857767 --exp_id DelayedCLS_Adv_100_316.22776601683796_100_2022 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6634193 --exp_id DelayedCLS_Adv_100_316.22776601683796_100_2024 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8858473 --exp_id DelayedCLS_Adv_100_316.22776601683796_100_2013 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5824012 --exp_id DelayedCLS_Adv_100_63.09573444801943_100_2017 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1101258 --exp_id DelayedCLS_Adv_100_63.09573444801943_100_2020 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1003253 --exp_id DelayedCLS_Adv_100_63.09573444801943_100_2022 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4739558 --exp_id DelayedCLS_Adv_100_63.09573444801943_100_2024 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7311425 --exp_id DelayedCLS_Adv_100_63.09573444801943_100_2013 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2177749 --exp_id DelayedCLS_Adv_100_0.007943282347242814_100_2017 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3760580 --exp_id DelayedCLS_Adv_100_0.007943282347242814_100_2020 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3229253 --exp_id DelayedCLS_Adv_100_0.007943282347242814_100_2022 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5294600 --exp_id DelayedCLS_Adv_100_0.007943282347242814_100_2024 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7395535 --exp_id DelayedCLS_Adv_100_0.007943282347242814_100_2013 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8992643 --exp_id DelayedCLS_Adv_100_0.06309573444801933_100_2017 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2234542 --exp_id DelayedCLS_Adv_100_0.06309573444801933_100_2020 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6656625 --exp_id DelayedCLS_Adv_100_0.06309573444801933_100_2022 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9757310 --exp_id DelayedCLS_Adv_100_0.06309573444801933_100_2024 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7027576 --exp_id DelayedCLS_Adv_100_0.06309573444801933_100_2013 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1979544 --exp_id DelayedCLS_Adv_100_0.1_100_2017 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3818236 --exp_id DelayedCLS_Adv_100_0.1_100_2020 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4430756 --exp_id DelayedCLS_Adv_100_0.1_100_2022 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5985940 --exp_id DelayedCLS_Adv_100_0.1_100_2024 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7262423 --exp_id DelayedCLS_Adv_100_0.1_100_2013 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1324942 --exp_id DelayedCLS_Adv_100_0.12589254117941676_100_2017 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2414219 --exp_id DelayedCLS_Adv_100_0.12589254117941676_100_2020 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 51243 --exp_id DelayedCLS_Adv_100_0.12589254117941676_100_2022 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7847293 --exp_id DelayedCLS_Adv_100_0.12589254117941676_100_2024 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5757234 --exp_id DelayedCLS_Adv_100_0.12589254117941676_100_2013 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 429585 --exp_id DelayedCLS_Adv_100_199.5262314968883_100_2017 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3893323 --exp_id DelayedCLS_Adv_100_199.5262314968883_100_2020 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8391239 --exp_id DelayedCLS_Adv_100_199.5262314968883_100_2022 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 860756 --exp_id DelayedCLS_Adv_100_199.5262314968883_100_2024 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7727168 --exp_id DelayedCLS_Adv_100_199.5262314968883_100_2013 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6631091 --exp_id DelayedCLS_Adv_100_79.43282347242821_100_2017 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 533183 --exp_id DelayedCLS_Adv_100_79.43282347242821_100_2020 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8555175 --exp_id DelayedCLS_Adv_100_79.43282347242821_100_2022 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7527044 --exp_id DelayedCLS_Adv_100_79.43282347242821_100_2024 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8260419 --exp_id DelayedCLS_Adv_100_79.43282347242821_100_2013 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9278338 --exp_id DelayedCLS_Adv_100_0.0199526231496888_100_2017 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4664782 --exp_id DelayedCLS_Adv_100_0.0199526231496888_100_2020 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7701019 --exp_id DelayedCLS_Adv_100_0.0199526231496888_100_2022 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 291173 --exp_id DelayedCLS_Adv_100_0.0199526231496888_100_2024 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3817426 --exp_id DelayedCLS_Adv_100_0.0199526231496888_100_2013 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9940133 --exp_id DelayedCLS_Adv_100_0.001584893192461114_100_2017 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 838800 --exp_id DelayedCLS_Adv_100_0.001584893192461114_100_2020 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1035983 --exp_id DelayedCLS_Adv_100_0.001584893192461114_100_2022 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9708517 --exp_id DelayedCLS_Adv_100_0.001584893192461114_100_2024 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 180687 --exp_id DelayedCLS_Adv_100_0.001584893192461114_100_2013 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7190171 --exp_id DelayedCLS_Adv_100_0.05011872336272725_100_2017 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3511773 --exp_id DelayedCLS_Adv_100_0.05011872336272725_100_2020 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3790584 --exp_id DelayedCLS_Adv_100_0.05011872336272725_100_2022 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3277766 --exp_id DelayedCLS_Adv_100_0.05011872336272725_100_2024 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7961598 --exp_id DelayedCLS_Adv_100_0.05011872336272725_100_2013 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1167630 --exp_id DelayedCLS_Adv_100_1.584893192461114_100_2017 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7195972 --exp_id DelayedCLS_Adv_100_1.584893192461114_100_2020 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1197337 --exp_id DelayedCLS_Adv_100_1.584893192461114_100_2022 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1058742 --exp_id DelayedCLS_Adv_100_1.584893192461114_100_2024 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2671437 --exp_id DelayedCLS_Adv_100_1.584893192461114_100_2013 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4441522 --exp_id DelayedCLS_Adv_100_1.9952623149688808_100_2017 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5456947 --exp_id DelayedCLS_Adv_100_1.9952623149688808_100_2020 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3923987 --exp_id DelayedCLS_Adv_100_1.9952623149688808_100_2022 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4878440 --exp_id DelayedCLS_Adv_100_1.9952623149688808_100_2024 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6223466 --exp_id DelayedCLS_Adv_100_1.9952623149688808_100_2013 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9426240 --exp_id DelayedCLS_Adv_100_0.07943282347242818_100_2017 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6255545 --exp_id DelayedCLS_Adv_100_0.07943282347242818_100_2020 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 245335 --exp_id DelayedCLS_Adv_100_0.07943282347242818_100_2022 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 439578 --exp_id DelayedCLS_Adv_100_0.07943282347242818_100_2024 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8273779 --exp_id DelayedCLS_Adv_100_0.07943282347242818_100_2013 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2687636 --exp_id DelayedCLS_Adv_100_100.0_100_2017 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7220913 --exp_id DelayedCLS_Adv_100_100.0_100_2020 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2689991 --exp_id DelayedCLS_Adv_100_100.0_100_2022 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7998732 --exp_id DelayedCLS_Adv_100_100.0_100_2024 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5080391 --exp_id DelayedCLS_Adv_100_100.0_100_2013 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8954074 --exp_id DelayedCLS_Adv_100_1000.0_100_2017 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7267509 --exp_id DelayedCLS_Adv_100_1000.0_100_2020 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9871975 --exp_id DelayedCLS_Adv_100_1000.0_100_2022 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9012227 --exp_id DelayedCLS_Adv_100_1000.0_100_2024 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4090354 --exp_id DelayedCLS_Adv_100_1000.0_100_2013 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 615865 --exp_id DelayedCLS_Adv_100_0.0012589254117941675_100_2017 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2781510 --exp_id DelayedCLS_Adv_100_0.0012589254117941675_100_2020 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7543177 --exp_id DelayedCLS_Adv_100_0.0012589254117941675_100_2022 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7830813 --exp_id DelayedCLS_Adv_100_0.0012589254117941675_100_2024 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1075564 --exp_id DelayedCLS_Adv_100_0.0012589254117941675_100_2013 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9887162 --exp_id DelayedCLS_Adv_100_0.01_100_2017 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3556352 --exp_id DelayedCLS_Adv_100_0.01_100_2020 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8392750 --exp_id DelayedCLS_Adv_100_0.01_100_2022 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9060333 --exp_id DelayedCLS_Adv_100_0.01_100_2024 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4716674 --exp_id DelayedCLS_Adv_100_0.01_100_2013 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5803087 --exp_id DelayedCLS_Adv_100_501.18723362727246_100_2017 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6053862 --exp_id DelayedCLS_Adv_100_501.18723362727246_100_2020 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7967917 --exp_id DelayedCLS_Adv_100_501.18723362727246_100_2022 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8599248 --exp_id DelayedCLS_Adv_100_501.18723362727246_100_2024 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7896299 --exp_id DelayedCLS_Adv_100_501.18723362727246_100_2013 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5342038 --exp_id DelayedCLS_Adv_100_630.9573444801943_100_2017 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3442287 --exp_id DelayedCLS_Adv_100_630.9573444801943_100_2020 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2201277 --exp_id DelayedCLS_Adv_100_630.9573444801943_100_2022 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6403442 --exp_id DelayedCLS_Adv_100_630.9573444801943_100_2024 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3305855 --exp_id DelayedCLS_Adv_100_630.9573444801943_100_2013 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2290439 --exp_id DelayedCLS_Adv_100_0.001995262314968879_100_2017 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8419040 --exp_id DelayedCLS_Adv_100_0.001995262314968879_100_2020 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5741489 --exp_id DelayedCLS_Adv_100_0.001995262314968879_100_2022 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8565598 --exp_id DelayedCLS_Adv_100_0.001995262314968879_100_2024 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3721230 --exp_id DelayedCLS_Adv_100_0.001995262314968879_100_2013 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6232826 --exp_id DelayedCLS_Adv_100_251.18864315095823_100_2017 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2937915 --exp_id DelayedCLS_Adv_100_251.18864315095823_100_2020 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6469070 --exp_id DelayedCLS_Adv_100_251.18864315095823_100_2022 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4883318 --exp_id DelayedCLS_Adv_100_251.18864315095823_100_2024 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3603032 --exp_id DelayedCLS_Adv_100_251.18864315095823_100_2013 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4066453 --exp_id DelayedCLS_Adv_100_125.89254117941688_100_2017 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3747776 --exp_id DelayedCLS_Adv_100_125.89254117941688_100_2020 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3493696 --exp_id DelayedCLS_Adv_100_125.89254117941688_100_2022 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5522211 --exp_id DelayedCLS_Adv_100_125.89254117941688_100_2024 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8031108 --exp_id DelayedCLS_Adv_100_125.89254117941688_100_2013 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1994411 --exp_id DelayedCLS_Adv_100_0.001_100_2017 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2678885 --exp_id DelayedCLS_Adv_100_0.001_100_2020 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4007408 --exp_id DelayedCLS_Adv_100_0.001_100_2022 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6628874 --exp_id DelayedCLS_Adv_100_0.001_100_2024 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
python main.py --project_dir hypertune --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5141696 --exp_id DelayedCLS_Adv_100_0.001_100_2013 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --classification_head_update_frequency 100
sleep 2
