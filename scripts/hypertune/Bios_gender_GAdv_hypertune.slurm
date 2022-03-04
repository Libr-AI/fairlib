#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name=GAdv_hypertune

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/GAdv_hypertune_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/GAdv_hypertune_%J.out

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

python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8527325 --exp_id GAdv_hypertune_0.07943282347242814_1_0_0 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1997381 --exp_id GAdv_hypertune_0.07943282347242814_1_0_1 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1206388 --exp_id GAdv_hypertune_0.07943282347242814_1_0_2 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1781173 --exp_id GAdv_hypertune_0.07943282347242814_1_0_3 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9710113 --exp_id GAdv_hypertune_0.07943282347242814_1_0_4 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6459440 --exp_id GAdv_hypertune_0.15848931924611143_1_0_0 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1733938 --exp_id GAdv_hypertune_0.15848931924611143_1_0_1 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3681496 --exp_id GAdv_hypertune_0.15848931924611143_1_0_2 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1400310 --exp_id GAdv_hypertune_0.15848931924611143_1_0_3 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 682394 --exp_id GAdv_hypertune_0.15848931924611143_1_0_4 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 566857 --exp_id GAdv_hypertune_0.19952623149688797_1_0_0 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6237412 --exp_id GAdv_hypertune_0.19952623149688797_1_0_1 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4673376 --exp_id GAdv_hypertune_0.19952623149688797_1_0_2 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4338491 --exp_id GAdv_hypertune_0.19952623149688797_1_0_3 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8985179 --exp_id GAdv_hypertune_0.19952623149688797_1_0_4 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4319132 --exp_id GAdv_hypertune_0.25118864315095807_1_0_0 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7291778 --exp_id GAdv_hypertune_0.25118864315095807_1_0_1 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2272113 --exp_id GAdv_hypertune_0.25118864315095807_1_0_2 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5322710 --exp_id GAdv_hypertune_0.25118864315095807_1_0_3 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2005597 --exp_id GAdv_hypertune_0.25118864315095807_1_0_4 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8256941 --exp_id GAdv_hypertune_0.5011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 85009 --exp_id GAdv_hypertune_0.5011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3993282 --exp_id GAdv_hypertune_0.5011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5887251 --exp_id GAdv_hypertune_0.5011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6478413 --exp_id GAdv_hypertune_0.5011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 871521 --exp_id GAdv_hypertune_0.6309573444801934_1_0_0 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8195160 --exp_id GAdv_hypertune_0.6309573444801934_1_0_1 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9051462 --exp_id GAdv_hypertune_0.6309573444801934_1_0_2 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9218893 --exp_id GAdv_hypertune_0.6309573444801934_1_0_3 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6026838 --exp_id GAdv_hypertune_0.6309573444801934_1_0_4 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5243826 --exp_id GAdv_hypertune_0.3981071705534973_1_0_0 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5098455 --exp_id GAdv_hypertune_0.3981071705534973_1_0_1 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 129221 --exp_id GAdv_hypertune_0.3981071705534973_1_0_2 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8722856 --exp_id GAdv_hypertune_0.3981071705534973_1_0_3 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5152568 --exp_id GAdv_hypertune_0.3981071705534973_1_0_4 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4450096 --exp_id GAdv_hypertune_0.31622776601683794_1_0_0 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2517609 --exp_id GAdv_hypertune_0.31622776601683794_1_0_1 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5849124 --exp_id GAdv_hypertune_0.31622776601683794_1_0_2 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3420058 --exp_id GAdv_hypertune_0.31622776601683794_1_0_3 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3665898 --exp_id GAdv_hypertune_0.31622776601683794_1_0_4 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7488012 --exp_id GAdv_hypertune_0.7943282347242817_1_0_0 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8222834 --exp_id GAdv_hypertune_0.7943282347242817_1_0_1 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1345035 --exp_id GAdv_hypertune_0.7943282347242817_1_0_2 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8571087 --exp_id GAdv_hypertune_0.7943282347242817_1_0_3 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6235768 --exp_id GAdv_hypertune_0.7943282347242817_1_0_4 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 459617 --exp_id GAdv_hypertune_1.0_1_0_0 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8134994 --exp_id GAdv_hypertune_1.0_1_0_1 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1041674 --exp_id GAdv_hypertune_1.0_1_0_2 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9292542 --exp_id GAdv_hypertune_1.0_1_0_3 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5252446 --exp_id GAdv_hypertune_1.0_1_0_4 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8507650 --exp_id GAdv_hypertune_1.2589254117941675_1_0_0 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 942051 --exp_id GAdv_hypertune_1.2589254117941675_1_0_1 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6541115 --exp_id GAdv_hypertune_1.2589254117941675_1_0_2 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7367721 --exp_id GAdv_hypertune_1.2589254117941675_1_0_3 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 257825 --exp_id GAdv_hypertune_1.2589254117941675_1_0_4 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9741601 --exp_id GAdv_hypertune_2.5118864315095824_1_0_0 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7533271 --exp_id GAdv_hypertune_2.5118864315095824_1_0_1 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9448781 --exp_id GAdv_hypertune_2.5118864315095824_1_0_2 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2537548 --exp_id GAdv_hypertune_2.5118864315095824_1_0_3 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5730295 --exp_id GAdv_hypertune_2.5118864315095824_1_0_4 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9579896 --exp_id GAdv_hypertune_3.1622776601683795_1_0_0 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 170445 --exp_id GAdv_hypertune_3.1622776601683795_1_0_1 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3355869 --exp_id GAdv_hypertune_3.1622776601683795_1_0_2 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8139124 --exp_id GAdv_hypertune_3.1622776601683795_1_0_3 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7029919 --exp_id GAdv_hypertune_3.1622776601683795_1_0_4 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6903172 --exp_id GAdv_hypertune_5.011872336272725_1_0_0 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1878200 --exp_id GAdv_hypertune_5.011872336272725_1_0_1 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9377600 --exp_id GAdv_hypertune_5.011872336272725_1_0_2 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 220573 --exp_id GAdv_hypertune_5.011872336272725_1_0_3 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1754863 --exp_id GAdv_hypertune_5.011872336272725_1_0_4 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 683479 --exp_id GAdv_hypertune_6.309573444801936_1_0_0 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2296086 --exp_id GAdv_hypertune_6.309573444801936_1_0_1 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7062593 --exp_id GAdv_hypertune_6.309573444801936_1_0_2 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3592485 --exp_id GAdv_hypertune_6.309573444801936_1_0_3 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2385106 --exp_id GAdv_hypertune_6.309573444801936_1_0_4 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2580584 --exp_id GAdv_hypertune_7.943282347242821_1_0_0 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2709899 --exp_id GAdv_hypertune_7.943282347242821_1_0_1 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3805349 --exp_id GAdv_hypertune_7.943282347242821_1_0_2 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5090301 --exp_id GAdv_hypertune_7.943282347242821_1_0_3 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8470209 --exp_id GAdv_hypertune_7.943282347242821_1_0_4 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4863302 --exp_id GAdv_hypertune_10.0_1_0_0 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4055212 --exp_id GAdv_hypertune_10.0_1_0_1 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9065487 --exp_id GAdv_hypertune_10.0_1_0_2 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5802025 --exp_id GAdv_hypertune_10.0_1_0_3 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9179199 --exp_id GAdv_hypertune_10.0_1_0_4 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 743664 --exp_id GAdv_hypertune_12.589254117941675_1_0_0 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7025940 --exp_id GAdv_hypertune_12.589254117941675_1_0_1 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4903271 --exp_id GAdv_hypertune_12.589254117941675_1_0_2 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2076541 --exp_id GAdv_hypertune_12.589254117941675_1_0_3 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 654390 --exp_id GAdv_hypertune_12.589254117941675_1_0_4 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1346453 --exp_id GAdv_hypertune_15.848931924611142_1_0_0 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3060067 --exp_id GAdv_hypertune_15.848931924611142_1_0_1 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3417322 --exp_id GAdv_hypertune_15.848931924611142_1_0_2 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4319799 --exp_id GAdv_hypertune_15.848931924611142_1_0_3 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6271841 --exp_id GAdv_hypertune_15.848931924611142_1_0_4 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2545279 --exp_id GAdv_hypertune_19.952623149688808_1_0_0 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2743704 --exp_id GAdv_hypertune_19.952623149688808_1_0_1 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1274284 --exp_id GAdv_hypertune_19.952623149688808_1_0_2 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9368174 --exp_id GAdv_hypertune_19.952623149688808_1_0_3 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2743557 --exp_id GAdv_hypertune_19.952623149688808_1_0_4 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2704626 --exp_id GAdv_hypertune_25.11886431509582_1_0_0 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5084172 --exp_id GAdv_hypertune_25.11886431509582_1_0_1 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7558576 --exp_id GAdv_hypertune_25.11886431509582_1_0_2 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2717396 --exp_id GAdv_hypertune_25.11886431509582_1_0_3 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6349197 --exp_id GAdv_hypertune_25.11886431509582_1_0_4 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8191376 --exp_id GAdv_hypertune_31.622776601683793_1_0_0 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1722679 --exp_id GAdv_hypertune_31.622776601683793_1_0_1 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3561874 --exp_id GAdv_hypertune_31.622776601683793_1_0_2 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 338400 --exp_id GAdv_hypertune_31.622776601683793_1_0_3 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2209110 --exp_id GAdv_hypertune_31.622776601683793_1_0_4 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 134970 --exp_id GAdv_hypertune_0.015848931924611134_1_0_0 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9569530 --exp_id GAdv_hypertune_0.015848931924611134_1_0_1 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2110906 --exp_id GAdv_hypertune_0.015848931924611134_1_0_2 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 387123 --exp_id GAdv_hypertune_0.015848931924611134_1_0_3 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1820697 --exp_id GAdv_hypertune_0.015848931924611134_1_0_4 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9506615 --exp_id GAdv_hypertune_0.03162277660168379_1_0_0 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2970789 --exp_id GAdv_hypertune_0.03162277660168379_1_0_1 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5019548 --exp_id GAdv_hypertune_0.03162277660168379_1_0_2 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5167223 --exp_id GAdv_hypertune_0.03162277660168379_1_0_3 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2879331 --exp_id GAdv_hypertune_0.03162277660168379_1_0_4 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5239269 --exp_id GAdv_hypertune_0.039810717055349734_1_0_0 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2405465 --exp_id GAdv_hypertune_0.039810717055349734_1_0_1 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7163710 --exp_id GAdv_hypertune_0.039810717055349734_1_0_2 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9632570 --exp_id GAdv_hypertune_0.039810717055349734_1_0_3 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9070621 --exp_id GAdv_hypertune_0.039810717055349734_1_0_4 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2731784 --exp_id GAdv_hypertune_39.810717055349734_1_0_0 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7633599 --exp_id GAdv_hypertune_39.810717055349734_1_0_1 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7361758 --exp_id GAdv_hypertune_39.810717055349734_1_0_2 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1191425 --exp_id GAdv_hypertune_39.810717055349734_1_0_3 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4002934 --exp_id GAdv_hypertune_39.810717055349734_1_0_4 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4869770 --exp_id GAdv_hypertune_3.981071705534973_1_0_0 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1722765 --exp_id GAdv_hypertune_3.981071705534973_1_0_1 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3368267 --exp_id GAdv_hypertune_3.981071705534973_1_0_2 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9715396 --exp_id GAdv_hypertune_3.981071705534973_1_0_3 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3958253 --exp_id GAdv_hypertune_3.981071705534973_1_0_4 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2735773 --exp_id GAdv_hypertune_50.11872336272725_1_0_0 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6761334 --exp_id GAdv_hypertune_50.11872336272725_1_0_1 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2608161 --exp_id GAdv_hypertune_50.11872336272725_1_0_2 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2085894 --exp_id GAdv_hypertune_50.11872336272725_1_0_3 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3287854 --exp_id GAdv_hypertune_50.11872336272725_1_0_4 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8699158 --exp_id GAdv_hypertune_0.012589254117941675_1_0_0 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 386861 --exp_id GAdv_hypertune_0.012589254117941675_1_0_1 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8162284 --exp_id GAdv_hypertune_0.012589254117941675_1_0_2 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9292735 --exp_id GAdv_hypertune_0.012589254117941675_1_0_3 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6512556 --exp_id GAdv_hypertune_0.012589254117941675_1_0_4 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1436473 --exp_id GAdv_hypertune_63.095734448019364_1_0_0 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2687001 --exp_id GAdv_hypertune_63.095734448019364_1_0_1 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2699271 --exp_id GAdv_hypertune_63.095734448019364_1_0_2 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9048833 --exp_id GAdv_hypertune_63.095734448019364_1_0_3 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1257893 --exp_id GAdv_hypertune_63.095734448019364_1_0_4 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4205596 --exp_id GAdv_hypertune_0.06309573444801933_1_0_0 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1183050 --exp_id GAdv_hypertune_0.06309573444801933_1_0_1 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8736921 --exp_id GAdv_hypertune_0.06309573444801933_1_0_2 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2228290 --exp_id GAdv_hypertune_0.06309573444801933_1_0_3 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2938774 --exp_id GAdv_hypertune_0.06309573444801933_1_0_4 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3243206 --exp_id GAdv_hypertune_0.1_1_0_0 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6418671 --exp_id GAdv_hypertune_0.1_1_0_1 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4233523 --exp_id GAdv_hypertune_0.1_1_0_2 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5323154 --exp_id GAdv_hypertune_0.1_1_0_3 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3520999 --exp_id GAdv_hypertune_0.1_1_0_4 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9508476 --exp_id GAdv_hypertune_0.12589254117941676_1_0_0 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5421586 --exp_id GAdv_hypertune_0.12589254117941676_1_0_1 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3058406 --exp_id GAdv_hypertune_0.12589254117941676_1_0_2 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8208282 --exp_id GAdv_hypertune_0.12589254117941676_1_0_3 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7106373 --exp_id GAdv_hypertune_0.12589254117941676_1_0_4 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1581623 --exp_id GAdv_hypertune_79.43282347242821_1_0_0 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4193945 --exp_id GAdv_hypertune_79.43282347242821_1_0_1 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 278030 --exp_id GAdv_hypertune_79.43282347242821_1_0_2 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5277394 --exp_id GAdv_hypertune_79.43282347242821_1_0_3 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9102058 --exp_id GAdv_hypertune_79.43282347242821_1_0_4 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2632231 --exp_id GAdv_hypertune_0.0199526231496888_1_0_0 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 589333 --exp_id GAdv_hypertune_0.0199526231496888_1_0_1 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7137748 --exp_id GAdv_hypertune_0.0199526231496888_1_0_2 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9796162 --exp_id GAdv_hypertune_0.0199526231496888_1_0_3 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 740704 --exp_id GAdv_hypertune_0.0199526231496888_1_0_4 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2809241 --exp_id GAdv_hypertune_0.05011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5055842 --exp_id GAdv_hypertune_0.05011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 656614 --exp_id GAdv_hypertune_0.05011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8298913 --exp_id GAdv_hypertune_0.05011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 171053 --exp_id GAdv_hypertune_0.05011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5631509 --exp_id GAdv_hypertune_1.584893192461114_1_0_0 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7538407 --exp_id GAdv_hypertune_1.584893192461114_1_0_1 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3297353 --exp_id GAdv_hypertune_1.584893192461114_1_0_2 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5754373 --exp_id GAdv_hypertune_1.584893192461114_1_0_3 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 144776 --exp_id GAdv_hypertune_1.584893192461114_1_0_4 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 656399 --exp_id GAdv_hypertune_1.9952623149688808_1_0_0 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4325428 --exp_id GAdv_hypertune_1.9952623149688808_1_0_1 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6638664 --exp_id GAdv_hypertune_1.9952623149688808_1_0_2 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9158991 --exp_id GAdv_hypertune_1.9952623149688808_1_0_3 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6264738 --exp_id GAdv_hypertune_1.9952623149688808_1_0_4 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4917825 --exp_id GAdv_hypertune_100.0_1_0_0 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8114815 --exp_id GAdv_hypertune_100.0_1_0_1 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6280829 --exp_id GAdv_hypertune_100.0_1_0_2 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6084954 --exp_id GAdv_hypertune_100.0_1_0_3 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7324867 --exp_id GAdv_hypertune_100.0_1_0_4 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4095521 --exp_id GAdv_hypertune_0.01_1_0_0 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7958472 --exp_id GAdv_hypertune_0.01_1_0_1 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3296084 --exp_id GAdv_hypertune_0.01_1_0_2 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1486932 --exp_id GAdv_hypertune_0.01_1_0_3 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2369659 --exp_id GAdv_hypertune_0.01_1_0_4 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7508899 --exp_id GAdv_hypertune_0.025118864315095794_1_0_0 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5792227 --exp_id GAdv_hypertune_0.025118864315095794_1_0_1 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9892551 --exp_id GAdv_hypertune_0.025118864315095794_1_0_2 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8686199 --exp_id GAdv_hypertune_0.025118864315095794_1_0_3 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 433895 --exp_id GAdv_hypertune_0.025118864315095794_1_0_4 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 2 --epochs 20
sleep 2
