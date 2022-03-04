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

python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4701431 --exp_id GAdv_hypertune_0.07943282347242814_1_0_0 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4223850 --exp_id GAdv_hypertune_0.07943282347242814_1_0_1 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1846287 --exp_id GAdv_hypertune_0.07943282347242814_1_0_2 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4995911 --exp_id GAdv_hypertune_0.07943282347242814_1_0_3 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7150754 --exp_id GAdv_hypertune_0.07943282347242814_1_0_4 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3507377 --exp_id GAdv_hypertune_0.15848931924611143_1_0_0 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4549682 --exp_id GAdv_hypertune_0.15848931924611143_1_0_1 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1365774 --exp_id GAdv_hypertune_0.15848931924611143_1_0_2 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6669750 --exp_id GAdv_hypertune_0.15848931924611143_1_0_3 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5204330 --exp_id GAdv_hypertune_0.15848931924611143_1_0_4 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7950443 --exp_id GAdv_hypertune_0.19952623149688797_1_0_0 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5722496 --exp_id GAdv_hypertune_0.19952623149688797_1_0_1 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6344716 --exp_id GAdv_hypertune_0.19952623149688797_1_0_2 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6478160 --exp_id GAdv_hypertune_0.19952623149688797_1_0_3 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7476885 --exp_id GAdv_hypertune_0.19952623149688797_1_0_4 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6512070 --exp_id GAdv_hypertune_0.25118864315095807_1_0_0 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9945668 --exp_id GAdv_hypertune_0.25118864315095807_1_0_1 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1957355 --exp_id GAdv_hypertune_0.25118864315095807_1_0_2 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8418197 --exp_id GAdv_hypertune_0.25118864315095807_1_0_3 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6617062 --exp_id GAdv_hypertune_0.25118864315095807_1_0_4 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9054269 --exp_id GAdv_hypertune_0.5011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7391064 --exp_id GAdv_hypertune_0.5011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8850120 --exp_id GAdv_hypertune_0.5011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9141165 --exp_id GAdv_hypertune_0.5011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 730848 --exp_id GAdv_hypertune_0.5011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 48847 --exp_id GAdv_hypertune_0.6309573444801934_1_0_0 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7846712 --exp_id GAdv_hypertune_0.6309573444801934_1_0_1 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6621282 --exp_id GAdv_hypertune_0.6309573444801934_1_0_2 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2979545 --exp_id GAdv_hypertune_0.6309573444801934_1_0_3 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1722959 --exp_id GAdv_hypertune_0.6309573444801934_1_0_4 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2169467 --exp_id GAdv_hypertune_0.3981071705534973_1_0_0 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2386633 --exp_id GAdv_hypertune_0.3981071705534973_1_0_1 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 424835 --exp_id GAdv_hypertune_0.3981071705534973_1_0_2 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2443402 --exp_id GAdv_hypertune_0.3981071705534973_1_0_3 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8645522 --exp_id GAdv_hypertune_0.3981071705534973_1_0_4 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5193091 --exp_id GAdv_hypertune_0.31622776601683794_1_0_0 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4742332 --exp_id GAdv_hypertune_0.31622776601683794_1_0_1 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8748350 --exp_id GAdv_hypertune_0.31622776601683794_1_0_2 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2408436 --exp_id GAdv_hypertune_0.31622776601683794_1_0_3 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5393476 --exp_id GAdv_hypertune_0.31622776601683794_1_0_4 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8289638 --exp_id GAdv_hypertune_0.7943282347242817_1_0_0 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2131460 --exp_id GAdv_hypertune_0.7943282347242817_1_0_1 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4234144 --exp_id GAdv_hypertune_0.7943282347242817_1_0_2 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3566615 --exp_id GAdv_hypertune_0.7943282347242817_1_0_3 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1447240 --exp_id GAdv_hypertune_0.7943282347242817_1_0_4 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3728212 --exp_id GAdv_hypertune_1.0_1_0_0 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6142229 --exp_id GAdv_hypertune_1.0_1_0_1 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5054422 --exp_id GAdv_hypertune_1.0_1_0_2 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1095483 --exp_id GAdv_hypertune_1.0_1_0_3 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3683969 --exp_id GAdv_hypertune_1.0_1_0_4 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5167127 --exp_id GAdv_hypertune_1.2589254117941675_1_0_0 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8033303 --exp_id GAdv_hypertune_1.2589254117941675_1_0_1 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3282439 --exp_id GAdv_hypertune_1.2589254117941675_1_0_2 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2577075 --exp_id GAdv_hypertune_1.2589254117941675_1_0_3 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2170094 --exp_id GAdv_hypertune_1.2589254117941675_1_0_4 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 609427 --exp_id GAdv_hypertune_2.5118864315095824_1_0_0 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5236509 --exp_id GAdv_hypertune_2.5118864315095824_1_0_1 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3545230 --exp_id GAdv_hypertune_2.5118864315095824_1_0_2 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6918416 --exp_id GAdv_hypertune_2.5118864315095824_1_0_3 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2459219 --exp_id GAdv_hypertune_2.5118864315095824_1_0_4 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4837496 --exp_id GAdv_hypertune_3.1622776601683795_1_0_0 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6965405 --exp_id GAdv_hypertune_3.1622776601683795_1_0_1 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3332119 --exp_id GAdv_hypertune_3.1622776601683795_1_0_2 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5931899 --exp_id GAdv_hypertune_3.1622776601683795_1_0_3 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6568568 --exp_id GAdv_hypertune_3.1622776601683795_1_0_4 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 795087 --exp_id GAdv_hypertune_5.011872336272725_1_0_0 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 371110 --exp_id GAdv_hypertune_5.011872336272725_1_0_1 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3969731 --exp_id GAdv_hypertune_5.011872336272725_1_0_2 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5638092 --exp_id GAdv_hypertune_5.011872336272725_1_0_3 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5763974 --exp_id GAdv_hypertune_5.011872336272725_1_0_4 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7269569 --exp_id GAdv_hypertune_6.309573444801936_1_0_0 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 85293 --exp_id GAdv_hypertune_6.309573444801936_1_0_1 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8914457 --exp_id GAdv_hypertune_6.309573444801936_1_0_2 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1702757 --exp_id GAdv_hypertune_6.309573444801936_1_0_3 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5941345 --exp_id GAdv_hypertune_6.309573444801936_1_0_4 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9339779 --exp_id GAdv_hypertune_7.943282347242821_1_0_0 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3907267 --exp_id GAdv_hypertune_7.943282347242821_1_0_1 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5582821 --exp_id GAdv_hypertune_7.943282347242821_1_0_2 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7414303 --exp_id GAdv_hypertune_7.943282347242821_1_0_3 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5479913 --exp_id GAdv_hypertune_7.943282347242821_1_0_4 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6328641 --exp_id GAdv_hypertune_10.0_1_0_0 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6399406 --exp_id GAdv_hypertune_10.0_1_0_1 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1511827 --exp_id GAdv_hypertune_10.0_1_0_2 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9713439 --exp_id GAdv_hypertune_10.0_1_0_3 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 132760 --exp_id GAdv_hypertune_10.0_1_0_4 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7444146 --exp_id GAdv_hypertune_12.589254117941675_1_0_0 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9154734 --exp_id GAdv_hypertune_12.589254117941675_1_0_1 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 638501 --exp_id GAdv_hypertune_12.589254117941675_1_0_2 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3131021 --exp_id GAdv_hypertune_12.589254117941675_1_0_3 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7460712 --exp_id GAdv_hypertune_12.589254117941675_1_0_4 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3870941 --exp_id GAdv_hypertune_15.848931924611142_1_0_0 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5925746 --exp_id GAdv_hypertune_15.848931924611142_1_0_1 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 617645 --exp_id GAdv_hypertune_15.848931924611142_1_0_2 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6768750 --exp_id GAdv_hypertune_15.848931924611142_1_0_3 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4457540 --exp_id GAdv_hypertune_15.848931924611142_1_0_4 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9423551 --exp_id GAdv_hypertune_19.952623149688808_1_0_0 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9808569 --exp_id GAdv_hypertune_19.952623149688808_1_0_1 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1504091 --exp_id GAdv_hypertune_19.952623149688808_1_0_2 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4196961 --exp_id GAdv_hypertune_19.952623149688808_1_0_3 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9310593 --exp_id GAdv_hypertune_19.952623149688808_1_0_4 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8511228 --exp_id GAdv_hypertune_25.11886431509582_1_0_0 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7155276 --exp_id GAdv_hypertune_25.11886431509582_1_0_1 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1115921 --exp_id GAdv_hypertune_25.11886431509582_1_0_2 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3408634 --exp_id GAdv_hypertune_25.11886431509582_1_0_3 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8617883 --exp_id GAdv_hypertune_25.11886431509582_1_0_4 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8263404 --exp_id GAdv_hypertune_31.622776601683793_1_0_0 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4514963 --exp_id GAdv_hypertune_31.622776601683793_1_0_1 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5127994 --exp_id GAdv_hypertune_31.622776601683793_1_0_2 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9775075 --exp_id GAdv_hypertune_31.622776601683793_1_0_3 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6782344 --exp_id GAdv_hypertune_31.622776601683793_1_0_4 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9335493 --exp_id GAdv_hypertune_0.015848931924611134_1_0_0 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4571843 --exp_id GAdv_hypertune_0.015848931924611134_1_0_1 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3538996 --exp_id GAdv_hypertune_0.015848931924611134_1_0_2 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5671004 --exp_id GAdv_hypertune_0.015848931924611134_1_0_3 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7610073 --exp_id GAdv_hypertune_0.015848931924611134_1_0_4 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5949371 --exp_id GAdv_hypertune_0.03162277660168379_1_0_0 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 213498 --exp_id GAdv_hypertune_0.03162277660168379_1_0_1 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9128246 --exp_id GAdv_hypertune_0.03162277660168379_1_0_2 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9301719 --exp_id GAdv_hypertune_0.03162277660168379_1_0_3 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5219683 --exp_id GAdv_hypertune_0.03162277660168379_1_0_4 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3376250 --exp_id GAdv_hypertune_0.039810717055349734_1_0_0 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2855226 --exp_id GAdv_hypertune_0.039810717055349734_1_0_1 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3226887 --exp_id GAdv_hypertune_0.039810717055349734_1_0_2 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1423594 --exp_id GAdv_hypertune_0.039810717055349734_1_0_3 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6538831 --exp_id GAdv_hypertune_0.039810717055349734_1_0_4 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6299686 --exp_id GAdv_hypertune_39.810717055349734_1_0_0 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 689606 --exp_id GAdv_hypertune_39.810717055349734_1_0_1 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2016815 --exp_id GAdv_hypertune_39.810717055349734_1_0_2 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 608394 --exp_id GAdv_hypertune_39.810717055349734_1_0_3 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5777782 --exp_id GAdv_hypertune_39.810717055349734_1_0_4 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 160746 --exp_id GAdv_hypertune_3.981071705534973_1_0_0 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7470794 --exp_id GAdv_hypertune_3.981071705534973_1_0_1 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6893549 --exp_id GAdv_hypertune_3.981071705534973_1_0_2 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2324156 --exp_id GAdv_hypertune_3.981071705534973_1_0_3 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5794081 --exp_id GAdv_hypertune_3.981071705534973_1_0_4 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7179008 --exp_id GAdv_hypertune_50.11872336272725_1_0_0 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8656696 --exp_id GAdv_hypertune_50.11872336272725_1_0_1 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2146019 --exp_id GAdv_hypertune_50.11872336272725_1_0_2 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4428309 --exp_id GAdv_hypertune_50.11872336272725_1_0_3 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2185119 --exp_id GAdv_hypertune_50.11872336272725_1_0_4 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1262456 --exp_id GAdv_hypertune_0.012589254117941675_1_0_0 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8010503 --exp_id GAdv_hypertune_0.012589254117941675_1_0_1 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7200757 --exp_id GAdv_hypertune_0.012589254117941675_1_0_2 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1471201 --exp_id GAdv_hypertune_0.012589254117941675_1_0_3 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5970954 --exp_id GAdv_hypertune_0.012589254117941675_1_0_4 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7074297 --exp_id GAdv_hypertune_63.095734448019364_1_0_0 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5721399 --exp_id GAdv_hypertune_63.095734448019364_1_0_1 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8594570 --exp_id GAdv_hypertune_63.095734448019364_1_0_2 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6577218 --exp_id GAdv_hypertune_63.095734448019364_1_0_3 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2130672 --exp_id GAdv_hypertune_63.095734448019364_1_0_4 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1311131 --exp_id GAdv_hypertune_0.06309573444801933_1_0_0 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9300253 --exp_id GAdv_hypertune_0.06309573444801933_1_0_1 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9165651 --exp_id GAdv_hypertune_0.06309573444801933_1_0_2 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6567285 --exp_id GAdv_hypertune_0.06309573444801933_1_0_3 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2463376 --exp_id GAdv_hypertune_0.06309573444801933_1_0_4 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2429436 --exp_id GAdv_hypertune_0.1_1_0_0 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3030920 --exp_id GAdv_hypertune_0.1_1_0_1 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6597670 --exp_id GAdv_hypertune_0.1_1_0_2 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1223255 --exp_id GAdv_hypertune_0.1_1_0_3 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4686976 --exp_id GAdv_hypertune_0.1_1_0_4 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4532585 --exp_id GAdv_hypertune_0.12589254117941676_1_0_0 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6008251 --exp_id GAdv_hypertune_0.12589254117941676_1_0_1 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5711306 --exp_id GAdv_hypertune_0.12589254117941676_1_0_2 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2102358 --exp_id GAdv_hypertune_0.12589254117941676_1_0_3 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7370707 --exp_id GAdv_hypertune_0.12589254117941676_1_0_4 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4703302 --exp_id GAdv_hypertune_79.43282347242821_1_0_0 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5114651 --exp_id GAdv_hypertune_79.43282347242821_1_0_1 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3581253 --exp_id GAdv_hypertune_79.43282347242821_1_0_2 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4298634 --exp_id GAdv_hypertune_79.43282347242821_1_0_3 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 489712 --exp_id GAdv_hypertune_79.43282347242821_1_0_4 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1579867 --exp_id GAdv_hypertune_0.0199526231496888_1_0_0 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1749660 --exp_id GAdv_hypertune_0.0199526231496888_1_0_1 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3845569 --exp_id GAdv_hypertune_0.0199526231496888_1_0_2 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9539991 --exp_id GAdv_hypertune_0.0199526231496888_1_0_3 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3255599 --exp_id GAdv_hypertune_0.0199526231496888_1_0_4 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7151276 --exp_id GAdv_hypertune_0.05011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4899582 --exp_id GAdv_hypertune_0.05011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7604603 --exp_id GAdv_hypertune_0.05011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 517545 --exp_id GAdv_hypertune_0.05011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 492643 --exp_id GAdv_hypertune_0.05011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3553638 --exp_id GAdv_hypertune_1.584893192461114_1_0_0 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3037645 --exp_id GAdv_hypertune_1.584893192461114_1_0_1 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1702575 --exp_id GAdv_hypertune_1.584893192461114_1_0_2 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7548736 --exp_id GAdv_hypertune_1.584893192461114_1_0_3 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4688786 --exp_id GAdv_hypertune_1.584893192461114_1_0_4 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8363713 --exp_id GAdv_hypertune_1.9952623149688808_1_0_0 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5325307 --exp_id GAdv_hypertune_1.9952623149688808_1_0_1 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 73992 --exp_id GAdv_hypertune_1.9952623149688808_1_0_2 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4469155 --exp_id GAdv_hypertune_1.9952623149688808_1_0_3 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2663002 --exp_id GAdv_hypertune_1.9952623149688808_1_0_4 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9285179 --exp_id GAdv_hypertune_100.0_1_0_0 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1488599 --exp_id GAdv_hypertune_100.0_1_0_1 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9100777 --exp_id GAdv_hypertune_100.0_1_0_2 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8735901 --exp_id GAdv_hypertune_100.0_1_0_3 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7870406 --exp_id GAdv_hypertune_100.0_1_0_4 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 775101 --exp_id GAdv_hypertune_0.01_1_0_0 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7469751 --exp_id GAdv_hypertune_0.01_1_0_1 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3816882 --exp_id GAdv_hypertune_0.01_1_0_2 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9924735 --exp_id GAdv_hypertune_0.01_1_0_3 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1361252 --exp_id GAdv_hypertune_0.01_1_0_4 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7664832 --exp_id GAdv_hypertune_0.025118864315095794_1_0_0 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 530124 --exp_id GAdv_hypertune_0.025118864315095794_1_0_1 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 297754 --exp_id GAdv_hypertune_0.025118864315095794_1_0_2 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7857407 --exp_id GAdv_hypertune_0.025118864315095794_1_0_3 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_both --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3219892 --exp_id GAdv_hypertune_0.025118864315095794_1_0_4 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_gated --adv_num_classes 4 --epochs 20
sleep 2
