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

python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6397784 --exp_id Adv_hypertune_0.07943282347242814_1_0_0 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5347435 --exp_id Adv_hypertune_0.07943282347242814_1_0_1 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8960051 --exp_id Adv_hypertune_0.07943282347242814_1_0_2 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 567701 --exp_id Adv_hypertune_0.07943282347242814_1_0_3 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 948091 --exp_id Adv_hypertune_0.07943282347242814_1_0_4 --adv_debiasing --adv_lambda 0.07943282347242814 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8771279 --exp_id Adv_hypertune_0.15848931924611143_1_0_0 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1439447 --exp_id Adv_hypertune_0.15848931924611143_1_0_1 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2650758 --exp_id Adv_hypertune_0.15848931924611143_1_0_2 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3203170 --exp_id Adv_hypertune_0.15848931924611143_1_0_3 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5406441 --exp_id Adv_hypertune_0.15848931924611143_1_0_4 --adv_debiasing --adv_lambda 0.15848931924611143 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7020442 --exp_id Adv_hypertune_0.19952623149688797_1_0_0 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8030469 --exp_id Adv_hypertune_0.19952623149688797_1_0_1 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6079886 --exp_id Adv_hypertune_0.19952623149688797_1_0_2 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9436476 --exp_id Adv_hypertune_0.19952623149688797_1_0_3 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2244986 --exp_id Adv_hypertune_0.19952623149688797_1_0_4 --adv_debiasing --adv_lambda 0.19952623149688797 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5055968 --exp_id Adv_hypertune_0.25118864315095807_1_0_0 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2750507 --exp_id Adv_hypertune_0.25118864315095807_1_0_1 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6662950 --exp_id Adv_hypertune_0.25118864315095807_1_0_2 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3138507 --exp_id Adv_hypertune_0.25118864315095807_1_0_3 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 914860 --exp_id Adv_hypertune_0.25118864315095807_1_0_4 --adv_debiasing --adv_lambda 0.25118864315095807 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 190962 --exp_id Adv_hypertune_0.5011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8958636 --exp_id Adv_hypertune_0.5011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5114418 --exp_id Adv_hypertune_0.5011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3521492 --exp_id Adv_hypertune_0.5011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8334722 --exp_id Adv_hypertune_0.5011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.5011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7104489 --exp_id Adv_hypertune_0.6309573444801934_1_0_0 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1002048 --exp_id Adv_hypertune_0.6309573444801934_1_0_1 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6468386 --exp_id Adv_hypertune_0.6309573444801934_1_0_2 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7300740 --exp_id Adv_hypertune_0.6309573444801934_1_0_3 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2234836 --exp_id Adv_hypertune_0.6309573444801934_1_0_4 --adv_debiasing --adv_lambda 0.6309573444801934 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8203129 --exp_id Adv_hypertune_0.3981071705534973_1_0_0 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2105600 --exp_id Adv_hypertune_0.3981071705534973_1_0_1 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7813805 --exp_id Adv_hypertune_0.3981071705534973_1_0_2 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5365723 --exp_id Adv_hypertune_0.3981071705534973_1_0_3 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8986951 --exp_id Adv_hypertune_0.3981071705534973_1_0_4 --adv_debiasing --adv_lambda 0.3981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8475902 --exp_id Adv_hypertune_0.31622776601683794_1_0_0 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1091093 --exp_id Adv_hypertune_0.31622776601683794_1_0_1 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3451919 --exp_id Adv_hypertune_0.31622776601683794_1_0_2 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5587536 --exp_id Adv_hypertune_0.31622776601683794_1_0_3 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5042789 --exp_id Adv_hypertune_0.31622776601683794_1_0_4 --adv_debiasing --adv_lambda 0.31622776601683794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5310232 --exp_id Adv_hypertune_0.7943282347242817_1_0_0 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2508191 --exp_id Adv_hypertune_0.7943282347242817_1_0_1 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2540324 --exp_id Adv_hypertune_0.7943282347242817_1_0_2 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5451219 --exp_id Adv_hypertune_0.7943282347242817_1_0_3 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5063394 --exp_id Adv_hypertune_0.7943282347242817_1_0_4 --adv_debiasing --adv_lambda 0.7943282347242817 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1380551 --exp_id Adv_hypertune_1.0_1_0_0 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7480186 --exp_id Adv_hypertune_1.0_1_0_1 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3756883 --exp_id Adv_hypertune_1.0_1_0_2 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3643670 --exp_id Adv_hypertune_1.0_1_0_3 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3615271 --exp_id Adv_hypertune_1.0_1_0_4 --adv_debiasing --adv_lambda 1.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8414677 --exp_id Adv_hypertune_1.2589254117941675_1_0_0 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6327346 --exp_id Adv_hypertune_1.2589254117941675_1_0_1 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1103120 --exp_id Adv_hypertune_1.2589254117941675_1_0_2 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5360509 --exp_id Adv_hypertune_1.2589254117941675_1_0_3 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2607101 --exp_id Adv_hypertune_1.2589254117941675_1_0_4 --adv_debiasing --adv_lambda 1.2589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3814010 --exp_id Adv_hypertune_2.5118864315095824_1_0_0 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1175381 --exp_id Adv_hypertune_2.5118864315095824_1_0_1 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5116425 --exp_id Adv_hypertune_2.5118864315095824_1_0_2 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8575085 --exp_id Adv_hypertune_2.5118864315095824_1_0_3 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6225576 --exp_id Adv_hypertune_2.5118864315095824_1_0_4 --adv_debiasing --adv_lambda 2.5118864315095824 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7046130 --exp_id Adv_hypertune_3.1622776601683795_1_0_0 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 353586 --exp_id Adv_hypertune_3.1622776601683795_1_0_1 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3779445 --exp_id Adv_hypertune_3.1622776601683795_1_0_2 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3371888 --exp_id Adv_hypertune_3.1622776601683795_1_0_3 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4707853 --exp_id Adv_hypertune_3.1622776601683795_1_0_4 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8006409 --exp_id Adv_hypertune_5.011872336272725_1_0_0 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8980313 --exp_id Adv_hypertune_5.011872336272725_1_0_1 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1444127 --exp_id Adv_hypertune_5.011872336272725_1_0_2 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7330093 --exp_id Adv_hypertune_5.011872336272725_1_0_3 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1785316 --exp_id Adv_hypertune_5.011872336272725_1_0_4 --adv_debiasing --adv_lambda 5.011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1562782 --exp_id Adv_hypertune_6.309573444801936_1_0_0 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9998985 --exp_id Adv_hypertune_6.309573444801936_1_0_1 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2135769 --exp_id Adv_hypertune_6.309573444801936_1_0_2 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2631909 --exp_id Adv_hypertune_6.309573444801936_1_0_3 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3936955 --exp_id Adv_hypertune_6.309573444801936_1_0_4 --adv_debiasing --adv_lambda 6.309573444801936 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9373351 --exp_id Adv_hypertune_7.943282347242821_1_0_0 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 220425 --exp_id Adv_hypertune_7.943282347242821_1_0_1 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1974449 --exp_id Adv_hypertune_7.943282347242821_1_0_2 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2989244 --exp_id Adv_hypertune_7.943282347242821_1_0_3 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6615742 --exp_id Adv_hypertune_7.943282347242821_1_0_4 --adv_debiasing --adv_lambda 7.943282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8223734 --exp_id Adv_hypertune_10.0_1_0_0 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6293079 --exp_id Adv_hypertune_10.0_1_0_1 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4315350 --exp_id Adv_hypertune_10.0_1_0_2 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7235977 --exp_id Adv_hypertune_10.0_1_0_3 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 829385 --exp_id Adv_hypertune_10.0_1_0_4 --adv_debiasing --adv_lambda 10.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2361251 --exp_id Adv_hypertune_12.589254117941675_1_0_0 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6491779 --exp_id Adv_hypertune_12.589254117941675_1_0_1 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1487386 --exp_id Adv_hypertune_12.589254117941675_1_0_2 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9760811 --exp_id Adv_hypertune_12.589254117941675_1_0_3 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2154127 --exp_id Adv_hypertune_12.589254117941675_1_0_4 --adv_debiasing --adv_lambda 12.589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4837272 --exp_id Adv_hypertune_15.848931924611142_1_0_0 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 733445 --exp_id Adv_hypertune_15.848931924611142_1_0_1 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7706758 --exp_id Adv_hypertune_15.848931924611142_1_0_2 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4643230 --exp_id Adv_hypertune_15.848931924611142_1_0_3 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8988658 --exp_id Adv_hypertune_15.848931924611142_1_0_4 --adv_debiasing --adv_lambda 15.848931924611142 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9206145 --exp_id Adv_hypertune_19.952623149688808_1_0_0 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2613642 --exp_id Adv_hypertune_19.952623149688808_1_0_1 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7573934 --exp_id Adv_hypertune_19.952623149688808_1_0_2 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4835491 --exp_id Adv_hypertune_19.952623149688808_1_0_3 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2183765 --exp_id Adv_hypertune_19.952623149688808_1_0_4 --adv_debiasing --adv_lambda 19.952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3642870 --exp_id Adv_hypertune_25.11886431509582_1_0_0 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7922033 --exp_id Adv_hypertune_25.11886431509582_1_0_1 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8022195 --exp_id Adv_hypertune_25.11886431509582_1_0_2 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1428941 --exp_id Adv_hypertune_25.11886431509582_1_0_3 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5383181 --exp_id Adv_hypertune_25.11886431509582_1_0_4 --adv_debiasing --adv_lambda 25.11886431509582 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2959045 --exp_id Adv_hypertune_31.622776601683793_1_0_0 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 645111 --exp_id Adv_hypertune_31.622776601683793_1_0_1 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5757703 --exp_id Adv_hypertune_31.622776601683793_1_0_2 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4146345 --exp_id Adv_hypertune_31.622776601683793_1_0_3 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 933266 --exp_id Adv_hypertune_31.622776601683793_1_0_4 --adv_debiasing --adv_lambda 31.622776601683793 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2499137 --exp_id Adv_hypertune_0.015848931924611134_1_0_0 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6084680 --exp_id Adv_hypertune_0.015848931924611134_1_0_1 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 115505 --exp_id Adv_hypertune_0.015848931924611134_1_0_2 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5182695 --exp_id Adv_hypertune_0.015848931924611134_1_0_3 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4374977 --exp_id Adv_hypertune_0.015848931924611134_1_0_4 --adv_debiasing --adv_lambda 0.015848931924611134 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1925050 --exp_id Adv_hypertune_0.03162277660168379_1_0_0 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 42859 --exp_id Adv_hypertune_0.03162277660168379_1_0_1 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4067067 --exp_id Adv_hypertune_0.03162277660168379_1_0_2 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3735934 --exp_id Adv_hypertune_0.03162277660168379_1_0_3 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6175847 --exp_id Adv_hypertune_0.03162277660168379_1_0_4 --adv_debiasing --adv_lambda 0.03162277660168379 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3445957 --exp_id Adv_hypertune_0.039810717055349734_1_0_0 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4986330 --exp_id Adv_hypertune_0.039810717055349734_1_0_1 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5075404 --exp_id Adv_hypertune_0.039810717055349734_1_0_2 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2497270 --exp_id Adv_hypertune_0.039810717055349734_1_0_3 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5967793 --exp_id Adv_hypertune_0.039810717055349734_1_0_4 --adv_debiasing --adv_lambda 0.039810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3706083 --exp_id Adv_hypertune_39.810717055349734_1_0_0 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2095001 --exp_id Adv_hypertune_39.810717055349734_1_0_1 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4886799 --exp_id Adv_hypertune_39.810717055349734_1_0_2 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2436612 --exp_id Adv_hypertune_39.810717055349734_1_0_3 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8724438 --exp_id Adv_hypertune_39.810717055349734_1_0_4 --adv_debiasing --adv_lambda 39.810717055349734 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5045102 --exp_id Adv_hypertune_3.981071705534973_1_0_0 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9258082 --exp_id Adv_hypertune_3.981071705534973_1_0_1 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7721502 --exp_id Adv_hypertune_3.981071705534973_1_0_2 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5909018 --exp_id Adv_hypertune_3.981071705534973_1_0_3 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6892719 --exp_id Adv_hypertune_3.981071705534973_1_0_4 --adv_debiasing --adv_lambda 3.981071705534973 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5142947 --exp_id Adv_hypertune_50.11872336272725_1_0_0 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5529897 --exp_id Adv_hypertune_50.11872336272725_1_0_1 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2063196 --exp_id Adv_hypertune_50.11872336272725_1_0_2 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 844625 --exp_id Adv_hypertune_50.11872336272725_1_0_3 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4942758 --exp_id Adv_hypertune_50.11872336272725_1_0_4 --adv_debiasing --adv_lambda 50.11872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2543523 --exp_id Adv_hypertune_0.012589254117941675_1_0_0 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9953368 --exp_id Adv_hypertune_0.012589254117941675_1_0_1 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6341104 --exp_id Adv_hypertune_0.012589254117941675_1_0_2 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3016777 --exp_id Adv_hypertune_0.012589254117941675_1_0_3 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2380562 --exp_id Adv_hypertune_0.012589254117941675_1_0_4 --adv_debiasing --adv_lambda 0.012589254117941675 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8903875 --exp_id Adv_hypertune_63.095734448019364_1_0_0 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6654462 --exp_id Adv_hypertune_63.095734448019364_1_0_1 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4112271 --exp_id Adv_hypertune_63.095734448019364_1_0_2 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5339399 --exp_id Adv_hypertune_63.095734448019364_1_0_3 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8803413 --exp_id Adv_hypertune_63.095734448019364_1_0_4 --adv_debiasing --adv_lambda 63.095734448019364 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7462900 --exp_id Adv_hypertune_0.06309573444801933_1_0_0 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8451168 --exp_id Adv_hypertune_0.06309573444801933_1_0_1 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3168905 --exp_id Adv_hypertune_0.06309573444801933_1_0_2 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 469554 --exp_id Adv_hypertune_0.06309573444801933_1_0_3 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9075770 --exp_id Adv_hypertune_0.06309573444801933_1_0_4 --adv_debiasing --adv_lambda 0.06309573444801933 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1587093 --exp_id Adv_hypertune_0.1_1_0_0 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5712363 --exp_id Adv_hypertune_0.1_1_0_1 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9178186 --exp_id Adv_hypertune_0.1_1_0_2 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 842368 --exp_id Adv_hypertune_0.1_1_0_3 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5891955 --exp_id Adv_hypertune_0.1_1_0_4 --adv_debiasing --adv_lambda 0.1 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3493544 --exp_id Adv_hypertune_0.12589254117941676_1_0_0 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 679960 --exp_id Adv_hypertune_0.12589254117941676_1_0_1 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5509224 --exp_id Adv_hypertune_0.12589254117941676_1_0_2 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1052206 --exp_id Adv_hypertune_0.12589254117941676_1_0_3 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2420441 --exp_id Adv_hypertune_0.12589254117941676_1_0_4 --adv_debiasing --adv_lambda 0.12589254117941676 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6718471 --exp_id Adv_hypertune_79.43282347242821_1_0_0 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9949828 --exp_id Adv_hypertune_79.43282347242821_1_0_1 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7050733 --exp_id Adv_hypertune_79.43282347242821_1_0_2 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4980463 --exp_id Adv_hypertune_79.43282347242821_1_0_3 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4617571 --exp_id Adv_hypertune_79.43282347242821_1_0_4 --adv_debiasing --adv_lambda 79.43282347242821 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3388568 --exp_id Adv_hypertune_0.0199526231496888_1_0_0 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7105103 --exp_id Adv_hypertune_0.0199526231496888_1_0_1 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1024064 --exp_id Adv_hypertune_0.0199526231496888_1_0_2 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6912351 --exp_id Adv_hypertune_0.0199526231496888_1_0_3 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3126323 --exp_id Adv_hypertune_0.0199526231496888_1_0_4 --adv_debiasing --adv_lambda 0.0199526231496888 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8847679 --exp_id Adv_hypertune_0.05011872336272725_1_0_0 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1932989 --exp_id Adv_hypertune_0.05011872336272725_1_0_1 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2120192 --exp_id Adv_hypertune_0.05011872336272725_1_0_2 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5687002 --exp_id Adv_hypertune_0.05011872336272725_1_0_3 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7583430 --exp_id Adv_hypertune_0.05011872336272725_1_0_4 --adv_debiasing --adv_lambda 0.05011872336272725 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5653473 --exp_id Adv_hypertune_1.584893192461114_1_0_0 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6116239 --exp_id Adv_hypertune_1.584893192461114_1_0_1 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6041511 --exp_id Adv_hypertune_1.584893192461114_1_0_2 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8534585 --exp_id Adv_hypertune_1.584893192461114_1_0_3 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6484511 --exp_id Adv_hypertune_1.584893192461114_1_0_4 --adv_debiasing --adv_lambda 1.584893192461114 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8229633 --exp_id Adv_hypertune_1.9952623149688808_1_0_0 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4553701 --exp_id Adv_hypertune_1.9952623149688808_1_0_1 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4758430 --exp_id Adv_hypertune_1.9952623149688808_1_0_2 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 805007 --exp_id Adv_hypertune_1.9952623149688808_1_0_3 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5907659 --exp_id Adv_hypertune_1.9952623149688808_1_0_4 --adv_debiasing --adv_lambda 1.9952623149688808 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5069832 --exp_id Adv_hypertune_100.0_1_0_0 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2028560 --exp_id Adv_hypertune_100.0_1_0_1 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6431419 --exp_id Adv_hypertune_100.0_1_0_2 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 721346 --exp_id Adv_hypertune_100.0_1_0_3 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9639535 --exp_id Adv_hypertune_100.0_1_0_4 --adv_debiasing --adv_lambda 100.0 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6708915 --exp_id Adv_hypertune_0.01_1_0_0 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1686138 --exp_id Adv_hypertune_0.01_1_0_1 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 415275 --exp_id Adv_hypertune_0.01_1_0_2 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9296837 --exp_id Adv_hypertune_0.01_1_0_3 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1470260 --exp_id Adv_hypertune_0.01_1_0_4 --adv_debiasing --adv_lambda 0.01 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1038519 --exp_id Adv_hypertune_0.025118864315095794_1_0_0 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2321410 --exp_id Adv_hypertune_0.025118864315095794_1_0_1 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7140813 --exp_id Adv_hypertune_0.025118864315095794_1_0_2 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5507755 --exp_id Adv_hypertune_0.025118864315095794_1_0_3 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
python main.py --project_dir hypertune4 --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6463031 --exp_id Adv_hypertune_0.025118864315095794_1_0_4 --adv_debiasing --adv_lambda 0.025118864315095794 --adv_num_subDiscriminator 1 --adv_diverse_lambda 0 --epochs_since_improvement 10 --adv_num_classes 2 --epochs 20
sleep 2
