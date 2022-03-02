#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=hypertune_DAdv_tunedLambda

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/hypertune_DAdv_tunedLambda_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/hypertune_DAdv_tunedLambda_%J.out

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

python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.12589254117941673_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.12589254117941673 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.12589254117941673_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.12589254117941673 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.12589254117941673_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.12589254117941673 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.12589254117941673_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.12589254117941673 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.12589254117941673_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.12589254117941673 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.19952623149688797_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.19952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.19952623149688797_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.19952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.19952623149688797_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.19952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.19952623149688797_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.19952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.19952623149688797_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.19952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.251188643150958_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.251188643150958 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.251188643150958_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.251188643150958 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.251188643150958_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.251188643150958 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.251188643150958_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.251188643150958 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.251188643150958_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.251188643150958 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.31622776601683794_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.31622776601683794 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.31622776601683794_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.31622776601683794 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.31622776601683794_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.31622776601683794 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.31622776601683794_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.31622776601683794 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.31622776601683794_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.31622776601683794 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.3981071705534973_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.3981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.3981071705534973_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.3981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.3981071705534973_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.3981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.3981071705534973_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.3981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.3981071705534973_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.3981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.5011872336272724_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.5011872336272724 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.5011872336272724_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.5011872336272724 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.5011872336272724_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.5011872336272724 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.5011872336272724_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.5011872336272724 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.5011872336272724_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.5011872336272724 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.6309573444801934_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.6309573444801934 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.6309573444801934_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.6309573444801934 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.6309573444801934_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.6309573444801934 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.6309573444801934_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.6309573444801934 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.6309573444801934_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.6309573444801934 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.7943282347242815_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.7943282347242815 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.7943282347242815_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.7943282347242815 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.7943282347242815_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.7943282347242815 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.7943282347242815_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.7943282347242815 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.7943282347242815_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.7943282347242815 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.0_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.0_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.0_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.0_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.0_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.9952623149688797_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.9952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.9952623149688797_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.9952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.9952623149688797_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.9952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.9952623149688797_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.9952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.9952623149688797_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.9952623149688797 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2.511886431509581_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2.511886431509581 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2.511886431509581_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2.511886431509581 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2.511886431509581_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2.511886431509581 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2.511886431509581_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2.511886431509581 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2.511886431509581_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2.511886431509581 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5.011872336272725_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5.011872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5.011872336272725_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5.011872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5.011872336272725_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5.011872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5.011872336272725_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5.011872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5.011872336272725_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5.011872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6.309573444801933_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6.309573444801933 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6.309573444801933_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6.309573444801933 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6.309573444801933_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6.309573444801933 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6.309573444801933_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6.309573444801933 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6.309573444801933_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6.309573444801933 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7.943282347242818_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7.943282347242818 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7.943282347242818_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7.943282347242818 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7.943282347242818_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7.943282347242818 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7.943282347242818_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7.943282347242818 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7.943282347242818_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7.943282347242818 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10.0_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10.0_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10.0_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10.0_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10.0_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12.589254117941675_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12.589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12.589254117941675_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12.589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12.589254117941675_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12.589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12.589254117941675_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12.589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12.589254117941675_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12.589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15.848931924611142_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15.848931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15.848931924611142_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15.848931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15.848931924611142_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15.848931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15.848931924611142_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15.848931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15.848931924611142_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15.848931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_398.1071705534973_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 398.1071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_398.1071705534973_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 398.1071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_398.1071705534973_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 398.1071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_398.1071705534973_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 398.1071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_398.1071705534973_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 398.1071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3981.0717055349774_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3981.0717055349774 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3981.0717055349774_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3981.0717055349774 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3981.0717055349774_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3981.0717055349774 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3981.0717055349774_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3981.0717055349774 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3981.0717055349774_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3981.0717055349774 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19.952623149688808_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19.952623149688808 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19.952623149688808_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19.952623149688808 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19.952623149688808_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19.952623149688808 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19.952623149688808_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19.952623149688808 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19.952623149688808_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19.952623149688808 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5011.872336272725_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5011.872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5011.872336272725_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5011.872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5011.872336272725_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5011.872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5011.872336272725_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5011.872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_5011.872336272725_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 5011.872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10000.0_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10000.0_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10000.0_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10000.0_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_10000.0_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 10000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25.11886431509582_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25.11886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25.11886431509582_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25.11886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25.11886431509582_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25.11886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25.11886431509582_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25.11886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25.11886431509582_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25.11886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_794.3282347242822_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 794.3282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_794.3282347242822_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 794.3282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_794.3282347242822_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 794.3282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_794.3282347242822_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 794.3282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_794.3282347242822_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 794.3282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7943.282347242822_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7943.282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7943.282347242822_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7943.282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7943.282347242822_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7943.282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7943.282347242822_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7943.282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_7943.282347242822_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 7943.282347242822 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_158.48931924611142_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 158.48931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_158.48931924611142_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 158.48931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_158.48931924611142_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 158.48931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_158.48931924611142_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 158.48931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_158.48931924611142_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 158.48931924611142 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31.622776601683793_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31.622776601683793 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31.622776601683793_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31.622776601683793 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31.622776601683793_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31.622776601683793 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31.622776601683793_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31.622776601683793 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31.622776601683793_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31.622776601683793 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25118.864315095823_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25118.864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25118.864315095823_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25118.864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25118.864315095823_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25118.864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25118.864315095823_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25118.864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_25118.864315095823_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 25118.864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31622.776601683792_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31622.776601683792 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31622.776601683792_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31622.776601683792 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31622.776601683792_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31622.776601683792 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31622.776601683792_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31622.776601683792 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_31622.776601683792_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 31622.776601683792 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100000.0_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100000.0_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100000.0_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100000.0_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100000.0_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6309.573444801943_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6309.573444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6309.573444801943_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6309.573444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6309.573444801943_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6309.573444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6309.573444801943_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6309.573444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_6309.573444801943_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 6309.573444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39.810717055349734_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39.810717055349734 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39.810717055349734_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39.810717055349734 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39.810717055349734_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39.810717055349734 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39.810717055349734_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39.810717055349734 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39.810717055349734_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39.810717055349734 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12589.254117941688_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12589.254117941688 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12589.254117941688_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12589.254117941688 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12589.254117941688_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12589.254117941688 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12589.254117941688_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12589.254117941688 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_12589.254117941688_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 12589.254117941688 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.981071705534973_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.981071705534973_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.981071705534973_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.981071705534973_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.981071705534973_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.981071705534973 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1584.893192461114_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1584.893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1584.893192461114_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1584.893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1584.893192461114_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1584.893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1584.893192461114_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1584.893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1584.893192461114_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1584.893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50.11872336272725_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50.11872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50.11872336272725_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50.11872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50.11872336272725_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50.11872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50.11872336272725_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50.11872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50.11872336272725_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50.11872336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_316.22776601683796_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 316.22776601683796 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_316.22776601683796_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 316.22776601683796 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_316.22776601683796_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 316.22776601683796 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_316.22776601683796_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 316.22776601683796 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_316.22776601683796_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 316.22776601683796 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63.095734448019364_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63.095734448019364 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63.095734448019364_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63.095734448019364 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63.095734448019364_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63.095734448019364 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63.095734448019364_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63.095734448019364 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63.095734448019364_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63.095734448019364 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.1_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.1_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.1_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.1_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.1_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.1 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.15848931924611134_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.15848931924611134 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.15848931924611134_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.15848931924611134 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.15848931924611134_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.15848931924611134 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.15848931924611134_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.15848931924611134 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_0.15848931924611134_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 0.15848931924611134 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50118.72336272725_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50118.72336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50118.72336272725_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50118.72336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50118.72336272725_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50118.72336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50118.72336272725_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50118.72336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_50118.72336272725_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 50118.72336272725 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_199.52623149688807_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 199.52623149688807 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_199.52623149688807_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 199.52623149688807 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_199.52623149688807_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 199.52623149688807 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_199.52623149688807_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 199.52623149688807 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_199.52623149688807_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 199.52623149688807 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79432.82347242821_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79432.82347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79432.82347242821_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79432.82347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79432.82347242821_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79432.82347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79432.82347242821_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79432.82347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79432.82347242821_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79432.82347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1995.2623149688789_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1995.2623149688789 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1995.2623149688789_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1995.2623149688789 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1995.2623149688789_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1995.2623149688789 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1995.2623149688789_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1995.2623149688789 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1995.2623149688789_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1995.2623149688789 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79.43282347242821_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79.43282347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79.43282347242821_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79.43282347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79.43282347242821_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79.43282347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79.43282347242821_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79.43282347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_79.43282347242821_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 79.43282347242821 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2511.886431509582_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2511.886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2511.886431509582_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2511.886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2511.886431509582_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2511.886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2511.886431509582_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2511.886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_2511.886431509582_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 2511.886431509582 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.2589254117941675_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.2589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.2589254117941675_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.2589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.2589254117941675_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.2589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.2589254117941675_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.2589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.2589254117941675_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.2589254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.584893192461114_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.584893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.584893192461114_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.584893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.584893192461114_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.584893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.584893192461114_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.584893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1.584893192461114_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1.584893192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3162.2776601683795_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3162.2776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3162.2776601683795_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3162.2776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3162.2776601683795_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3162.2776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3162.2776601683795_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3162.2776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3162.2776601683795_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3162.2776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100.0_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100.0_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100.0_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100.0_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_100.0_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 100.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39810.71705534977_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39810.71705534977 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39810.71705534977_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39810.71705534977 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39810.71705534977_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39810.71705534977 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39810.71705534977_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39810.71705534977 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_39810.71705534977_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 39810.71705534977 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1000.0_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1000.0_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1000.0_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1000.0_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1000.0_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1000.0 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15848.93192461114_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15848.93192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15848.93192461114_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15848.93192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15848.93192461114_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15848.93192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15848.93192461114_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15848.93192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_15848.93192461114_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 15848.93192461114 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1258.9254117941687_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1258.9254117941687 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1258.9254117941687_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1258.9254117941687 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1258.9254117941687_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1258.9254117941687 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1258.9254117941687_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1258.9254117941687 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_1258.9254117941687_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 1258.9254117941687 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.1622776601683795_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.1622776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.1622776601683795_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.1622776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.1622776601683795_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.1622776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.1622776601683795_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.1622776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_3.1622776601683795_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 3.1622776601683795 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19952.62314968883_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19952.62314968883 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19952.62314968883_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19952.62314968883 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19952.62314968883_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19952.62314968883 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19952.62314968883_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19952.62314968883 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_19952.62314968883_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 19952.62314968883 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_501.18723362727246_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 501.18723362727246 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_501.18723362727246_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 501.18723362727246 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_501.18723362727246_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 501.18723362727246 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_501.18723362727246_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 501.18723362727246 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_501.18723362727246_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 501.18723362727246 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_630.9573444801937_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 630.9573444801937 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_630.9573444801937_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 630.9573444801937 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_630.9573444801937_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 630.9573444801937 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_630.9573444801937_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 630.9573444801937 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_630.9573444801937_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 630.9573444801937 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63095.73444801943_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63095.73444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63095.73444801943_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63095.73444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63095.73444801943_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63095.73444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63095.73444801943_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63095.73444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_63095.73444801943_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 63095.73444801943 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_251.18864315095823_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 251.18864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_251.18864315095823_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 251.18864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_251.18864315095823_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 251.18864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_251.18864315095823_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 251.18864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_251.18864315095823_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 251.18864315095823 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_125.89254117941675_2017 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 125.89254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2020 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_125.89254117941675_2020 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 125.89254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2022 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_125.89254117941675_2022 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 125.89254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2024 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_125.89254117941675_2024 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 125.89254117941675 --epochs_since_improvement 10
sleep 2
python main.py --project_dir hypertune2 --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2013 --exp_id hypertune_DAdv_tunedLambda_3.1622776601683795_3_125.89254117941675_2013 --adv_debiasing --adv_lambda 3.1622776601683795 --adv_num_subDiscriminator 3 --adv_diverse_lambda 125.89254117941675 --epochs_since_improvement 10
sleep 2
