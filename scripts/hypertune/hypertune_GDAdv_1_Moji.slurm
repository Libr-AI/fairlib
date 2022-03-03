#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=hypertune_GDAdv_1

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/hypertune_GDAdv_1_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/hypertune_GDAdv_1_%J.out

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

python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.00630957344480193_3_0.1_2017 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.00630957344480193_3_0.1_2020 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.00630957344480193_3_0.1_2022 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.00630957344480193_3_0.1_2024 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.00630957344480193_3_0.1_2013 --adv_debiasing --adv_lambda 0.00630957344480193 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.15848931924611143_3_0.1_2017 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.15848931924611143_3_0.1_2020 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.15848931924611143_3_0.1_2022 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.15848931924611143_3_0.1_2024 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.15848931924611143_3_0.1_2013 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.19952623149688808_3_0.1_2017 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.19952623149688808_3_0.1_2020 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.19952623149688808_3_0.1_2022 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.19952623149688808_3_0.1_2024 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.19952623149688808_3_0.1_2013 --adv_debiasing --adv_lambda 0.19952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.25118864315095824_3_0.1_2017 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.25118864315095824_3_0.1_2020 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.25118864315095824_3_0.1_2022 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.25118864315095824_3_0.1_2024 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.25118864315095824_3_0.1_2013 --adv_debiasing --adv_lambda 0.25118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.31622776601683794_3_0.1_2017 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.31622776601683794_3_0.1_2020 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.31622776601683794_3_0.1_2022 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.31622776601683794_3_0.1_2024 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.31622776601683794_3_0.1_2013 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.3981071705534973_3_0.1_2017 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.3981071705534973_3_0.1_2020 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.3981071705534973_3_0.1_2022 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.3981071705534973_3_0.1_2024 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.3981071705534973_3_0.1_2013 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.5011872336272725_3_0.1_2017 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.5011872336272725_3_0.1_2020 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.5011872336272725_3_0.1_2022 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.5011872336272725_3_0.1_2024 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.5011872336272725_3_0.1_2013 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.6309573444801936_3_0.1_2017 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.6309573444801936_3_0.1_2020 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.6309573444801936_3_0.1_2022 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.6309573444801936_3_0.1_2024 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.6309573444801936_3_0.1_2013 --adv_debiasing --adv_lambda 0.6309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.7943282347242822_3_0.1_2017 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.7943282347242822_3_0.1_2020 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.7943282347242822_3_0.1_2022 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.7943282347242822_3_0.1_2024 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.7943282347242822_3_0.1_2013 --adv_debiasing --adv_lambda 0.7943282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_1.0_3_0.1_2017 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_1.0_3_0.1_2020 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_1.0_3_0.1_2022 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_1.0_3_0.1_2024 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_1.0_3_0.1_2013 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_1.2589254117941675_3_0.1_2017 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_1.2589254117941675_3_0.1_2020 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_1.2589254117941675_3_0.1_2022 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_1.2589254117941675_3_0.1_2024 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_1.2589254117941675_3_0.1_2013 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_2.5118864315095824_3_0.1_2017 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_2.5118864315095824_3_0.1_2020 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_2.5118864315095824_3_0.1_2022 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_2.5118864315095824_3_0.1_2024 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_2.5118864315095824_3_0.1_2013 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_3.1622776601683795_3_0.1_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_3.1622776601683795_3_0.1_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_3.1622776601683795_3_0.1_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_3.1622776601683795_3_0.1_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_3.1622776601683795_3_0.1_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_5.011872336272725_3_0.1_2017 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_5.011872336272725_3_0.1_2020 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_5.011872336272725_3_0.1_2022 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_5.011872336272725_3_0.1_2024 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_5.011872336272725_3_0.1_2013 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_6.309573444801936_3_0.1_2017 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_6.309573444801936_3_0.1_2020 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_6.309573444801936_3_0.1_2022 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_6.309573444801936_3_0.1_2024 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_6.309573444801936_3_0.1_2013 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_7.943282347242821_3_0.1_2017 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_7.943282347242821_3_0.1_2020 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_7.943282347242821_3_0.1_2022 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_7.943282347242821_3_0.1_2024 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_7.943282347242821_3_0.1_2013 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.003981071705534973_3_0.1_2017 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.003981071705534973_3_0.1_2020 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.003981071705534973_3_0.1_2022 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.003981071705534973_3_0.1_2024 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.003981071705534973_3_0.1_2013 --adv_debiasing --adv_lambda 0.003981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.025118864315095808_3_0.1_2017 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.025118864315095808_3_0.1_2020 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.025118864315095808_3_0.1_2022 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.025118864315095808_3_0.1_2024 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.025118864315095808_3_0.1_2013 --adv_debiasing --adv_lambda 0.025118864315095808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_10.0_3_0.1_2017 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_10.0_3_0.1_2020 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_10.0_3_0.1_2022 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_10.0_3_0.1_2024 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_10.0_3_0.1_2013 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_12.589254117941687_3_0.1_2017 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_12.589254117941687_3_0.1_2020 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_12.589254117941687_3_0.1_2022 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_12.589254117941687_3_0.1_2024 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_12.589254117941687_3_0.1_2013 --adv_debiasing --adv_lambda 12.589254117941687 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_15.848931924611142_3_0.1_2017 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_15.848931924611142_3_0.1_2020 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_15.848931924611142_3_0.1_2022 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_15.848931924611142_3_0.1_2024 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_15.848931924611142_3_0.1_2013 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_19.952623149688787_3_0.1_2017 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_19.952623149688787_3_0.1_2020 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_19.952623149688787_3_0.1_2022 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_19.952623149688787_3_0.1_2024 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_19.952623149688787_3_0.1_2013 --adv_debiasing --adv_lambda 19.952623149688787 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_398.1071705534977_3_0.1_2017 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_398.1071705534977_3_0.1_2020 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_398.1071705534977_3_0.1_2022 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_398.1071705534977_3_0.1_2024 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_398.1071705534977_3_0.1_2013 --adv_debiasing --adv_lambda 398.1071705534977 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.0025118864315095794_3_0.1_2017 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.0025118864315095794_3_0.1_2020 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.0025118864315095794_3_0.1_2022 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.0025118864315095794_3_0.1_2024 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.0025118864315095794_3_0.1_2013 --adv_debiasing --adv_lambda 0.0025118864315095794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_25.11886431509582_3_0.1_2017 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_25.11886431509582_3_0.1_2020 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_25.11886431509582_3_0.1_2022 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_25.11886431509582_3_0.1_2024 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_25.11886431509582_3_0.1_2013 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_794.3282347242822_3_0.1_2017 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_794.3282347242822_3_0.1_2020 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_794.3282347242822_3_0.1_2022 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_794.3282347242822_3_0.1_2024 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_794.3282347242822_3_0.1_2013 --adv_debiasing --adv_lambda 794.3282347242822 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_158.48931924611142_3_0.1_2017 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_158.48931924611142_3_0.1_2020 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_158.48931924611142_3_0.1_2022 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_158.48931924611142_3_0.1_2024 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_158.48931924611142_3_0.1_2013 --adv_debiasing --adv_lambda 158.48931924611142 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_31.622776601683793_3_0.1_2017 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_31.622776601683793_3_0.1_2020 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_31.622776601683793_3_0.1_2022 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_31.622776601683793_3_0.1_2024 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_31.622776601683793_3_0.1_2013 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.03162277660168379_3_0.1_2017 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.03162277660168379_3_0.1_2020 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.03162277660168379_3_0.1_2022 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.03162277660168379_3_0.1_2024 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.03162277660168379_3_0.1_2013 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.039810717055349734_3_0.1_2017 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.039810717055349734_3_0.1_2020 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.039810717055349734_3_0.1_2022 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.039810717055349734_3_0.1_2024 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.039810717055349734_3_0.1_2013 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_39.81071705534978_3_0.1_2017 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_39.81071705534978_3_0.1_2020 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_39.81071705534978_3_0.1_2022 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_39.81071705534978_3_0.1_2024 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_39.81071705534978_3_0.1_2013 --adv_debiasing --adv_lambda 39.81071705534978 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.005011872336272725_3_0.1_2017 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.005011872336272725_3_0.1_2020 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.005011872336272725_3_0.1_2022 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.005011872336272725_3_0.1_2024 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.005011872336272725_3_0.1_2013 --adv_debiasing --adv_lambda 0.005011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.01584893192461114_3_0.1_2017 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.01584893192461114_3_0.1_2020 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.01584893192461114_3_0.1_2022 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.01584893192461114_3_0.1_2024 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.01584893192461114_3_0.1_2013 --adv_debiasing --adv_lambda 0.01584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_3.981071705534973_3_0.1_2017 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_3.981071705534973_3_0.1_2020 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_3.981071705534973_3_0.1_2022 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_3.981071705534973_3_0.1_2024 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_3.981071705534973_3_0.1_2013 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_50.11872336272725_3_0.1_2017 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_50.11872336272725_3_0.1_2020 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_50.11872336272725_3_0.1_2022 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_50.11872336272725_3_0.1_2024 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_50.11872336272725_3_0.1_2013 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.0031622776601683794_3_0.1_2017 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.0031622776601683794_3_0.1_2020 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.0031622776601683794_3_0.1_2022 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.0031622776601683794_3_0.1_2024 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.0031622776601683794_3_0.1_2013 --adv_debiasing --adv_lambda 0.0031622776601683794 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.012589254117941675_3_0.1_2017 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.012589254117941675_3_0.1_2020 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.012589254117941675_3_0.1_2022 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.012589254117941675_3_0.1_2024 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.012589254117941675_3_0.1_2013 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_316.22776601683796_3_0.1_2017 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_316.22776601683796_3_0.1_2020 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_316.22776601683796_3_0.1_2022 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_316.22776601683796_3_0.1_2024 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_316.22776601683796_3_0.1_2013 --adv_debiasing --adv_lambda 316.22776601683796 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_63.09573444801943_3_0.1_2017 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_63.09573444801943_3_0.1_2020 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_63.09573444801943_3_0.1_2022 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_63.09573444801943_3_0.1_2024 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_63.09573444801943_3_0.1_2013 --adv_debiasing --adv_lambda 63.09573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.007943282347242814_3_0.1_2017 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.007943282347242814_3_0.1_2020 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.007943282347242814_3_0.1_2022 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.007943282347242814_3_0.1_2024 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.007943282347242814_3_0.1_2013 --adv_debiasing --adv_lambda 0.007943282347242814 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.06309573444801933_3_0.1_2017 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.06309573444801933_3_0.1_2020 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.06309573444801933_3_0.1_2022 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.06309573444801933_3_0.1_2024 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.06309573444801933_3_0.1_2013 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.1_3_0.1_2017 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.1_3_0.1_2020 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.1_3_0.1_2022 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.1_3_0.1_2024 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.1_3_0.1_2013 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.12589254117941676_3_0.1_2017 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.12589254117941676_3_0.1_2020 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.12589254117941676_3_0.1_2022 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.12589254117941676_3_0.1_2024 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.12589254117941676_3_0.1_2013 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_199.5262314968883_3_0.1_2017 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_199.5262314968883_3_0.1_2020 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_199.5262314968883_3_0.1_2022 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_199.5262314968883_3_0.1_2024 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_199.5262314968883_3_0.1_2013 --adv_debiasing --adv_lambda 199.5262314968883 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_79.43282347242821_3_0.1_2017 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_79.43282347242821_3_0.1_2020 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_79.43282347242821_3_0.1_2022 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_79.43282347242821_3_0.1_2024 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_79.43282347242821_3_0.1_2013 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.0199526231496888_3_0.1_2017 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.0199526231496888_3_0.1_2020 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.0199526231496888_3_0.1_2022 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.0199526231496888_3_0.1_2024 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.0199526231496888_3_0.1_2013 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.001584893192461114_3_0.1_2017 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.001584893192461114_3_0.1_2020 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.001584893192461114_3_0.1_2022 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.001584893192461114_3_0.1_2024 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.001584893192461114_3_0.1_2013 --adv_debiasing --adv_lambda 0.001584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.05011872336272725_3_0.1_2017 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.05011872336272725_3_0.1_2020 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.05011872336272725_3_0.1_2022 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.05011872336272725_3_0.1_2024 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.05011872336272725_3_0.1_2013 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_1.584893192461114_3_0.1_2017 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_1.584893192461114_3_0.1_2020 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_1.584893192461114_3_0.1_2022 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_1.584893192461114_3_0.1_2024 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_1.584893192461114_3_0.1_2013 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_1.9952623149688808_3_0.1_2017 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_1.9952623149688808_3_0.1_2020 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_1.9952623149688808_3_0.1_2022 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_1.9952623149688808_3_0.1_2024 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_1.9952623149688808_3_0.1_2013 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.07943282347242818_3_0.1_2017 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.07943282347242818_3_0.1_2020 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.07943282347242818_3_0.1_2022 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.07943282347242818_3_0.1_2024 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.07943282347242818_3_0.1_2013 --adv_debiasing --adv_lambda 0.07943282347242818 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_100.0_3_0.1_2017 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_100.0_3_0.1_2020 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_100.0_3_0.1_2022 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_100.0_3_0.1_2024 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_100.0_3_0.1_2013 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_1000.0_3_0.1_2017 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_1000.0_3_0.1_2020 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_1000.0_3_0.1_2022 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_1000.0_3_0.1_2024 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_1000.0_3_0.1_2013 --adv_debiasing --adv_lambda 1000.0 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.0012589254117941675_3_0.1_2017 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.0012589254117941675_3_0.1_2020 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.0012589254117941675_3_0.1_2022 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.0012589254117941675_3_0.1_2024 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.0012589254117941675_3_0.1_2013 --adv_debiasing --adv_lambda 0.0012589254117941675 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.01_3_0.1_2017 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.01_3_0.1_2020 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.01_3_0.1_2022 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.01_3_0.1_2024 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.01_3_0.1_2013 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_501.18723362727246_3_0.1_2017 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_501.18723362727246_3_0.1_2020 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_501.18723362727246_3_0.1_2022 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_501.18723362727246_3_0.1_2024 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_501.18723362727246_3_0.1_2013 --adv_debiasing --adv_lambda 501.18723362727246 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_630.9573444801943_3_0.1_2017 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_630.9573444801943_3_0.1_2020 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_630.9573444801943_3_0.1_2022 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_630.9573444801943_3_0.1_2024 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_630.9573444801943_3_0.1_2013 --adv_debiasing --adv_lambda 630.9573444801943 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.001995262314968879_3_0.1_2017 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.001995262314968879_3_0.1_2020 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.001995262314968879_3_0.1_2022 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.001995262314968879_3_0.1_2024 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.001995262314968879_3_0.1_2013 --adv_debiasing --adv_lambda 0.001995262314968879 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_251.18864315095823_3_0.1_2017 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_251.18864315095823_3_0.1_2020 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_251.18864315095823_3_0.1_2022 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_251.18864315095823_3_0.1_2024 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_251.18864315095823_3_0.1_2013 --adv_debiasing --adv_lambda 251.18864315095823 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_125.89254117941688_3_0.1_2017 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_125.89254117941688_3_0.1_2020 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_125.89254117941688_3_0.1_2022 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_125.89254117941688_3_0.1_2024 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_125.89254117941688_3_0.1_2013 --adv_debiasing --adv_lambda 125.89254117941688 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_GDAdv_1_0.001_3_0.1_2017 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_GDAdv_1_0.001_3_0.1_2020 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_GDAdv_1_0.001_3_0.1_2022 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_GDAdv_1_0.001_3_0.1_2024 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
python main.py --project_dir hypertune3 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_GDAdv_1_0.001_3_0.1_2013 --adv_debiasing --adv_lambda 0.001 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10 --adv_gated
sleep 2
