#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Moji_FSCL_EO

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Moji_FSCL_EO_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Moji_FSCL_EO_%J.out

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

python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6594083 --exp_id FSCL_EO_0.00630957344480193_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1614274 --exp_id FSCL_EO_0.15848931924611143_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3327136 --exp_id FSCL_EO_0.19952623149688808_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1377867 --exp_id FSCL_EO_0.25118864315095824_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6366556 --exp_id FSCL_EO_0.31622776601683794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2586328 --exp_id FSCL_EO_0.3981071705534973_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9732248 --exp_id FSCL_EO_0.5011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7772031 --exp_id FSCL_EO_0.6309573444801936_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3787967 --exp_id FSCL_EO_0.7943282347242822_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5846453 --exp_id FSCL_EO_1.0_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1143668 --exp_id FSCL_EO_1.2589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3439328 --exp_id FSCL_EO_2.5118864315095824_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 466644 --exp_id FSCL_EO_3.1622776601683795_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 136260 --exp_id FSCL_EO_5.011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8940003 --exp_id FSCL_EO_6.309573444801936_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 602986 --exp_id FSCL_EO_7.943282347242821_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4747047 --exp_id FSCL_EO_0.003981071705534973_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9629517 --exp_id FSCL_EO_0.025118864315095808_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4550638 --exp_id FSCL_EO_10.0_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1641878 --exp_id FSCL_EO_0.0025118864315095794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2167562 --exp_id FSCL_EO_0.03162277660168379_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 857190 --exp_id FSCL_EO_0.039810717055349734_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9901511 --exp_id FSCL_EO_0.005011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9637614 --exp_id FSCL_EO_0.01584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9820655 --exp_id FSCL_EO_3.981071705534973_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2053670 --exp_id FSCL_EO_0.0031622776601683794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3654773 --exp_id FSCL_EO_0.012589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7067769 --exp_id FSCL_EO_0.007943282347242814_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3505270 --exp_id FSCL_EO_0.06309573444801933_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8891508 --exp_id FSCL_EO_0.1_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7239191 --exp_id FSCL_EO_0.12589254117941676_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8071907 --exp_id FSCL_EO_0.0199526231496888_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4909809 --exp_id FSCL_EO_0.001584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4295578 --exp_id FSCL_EO_0.05011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4320488 --exp_id FSCL_EO_1.584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3881673 --exp_id FSCL_EO_1.9952623149688808_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1305642 --exp_id FSCL_EO_0.07943282347242818_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8908416 --exp_id FSCL_EO_0.0012589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 249416 --exp_id FSCL_EO_0.01_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5089325 --exp_id FSCL_EO_0.001995262314968879_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2374289 --exp_id FSCL_EO_0.001_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5089585 --exp_id FSCL_EO_0.00630957344480193_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1512185 --exp_id FSCL_EO_0.15848931924611143_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 777658 --exp_id FSCL_EO_0.19952623149688808_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6063849 --exp_id FSCL_EO_0.25118864315095824_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4005066 --exp_id FSCL_EO_0.31622776601683794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8903232 --exp_id FSCL_EO_0.3981071705534973_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5301390 --exp_id FSCL_EO_0.5011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7327335 --exp_id FSCL_EO_0.6309573444801936_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5933503 --exp_id FSCL_EO_0.7943282347242822_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3571342 --exp_id FSCL_EO_1.0_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1122922 --exp_id FSCL_EO_1.2589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6215701 --exp_id FSCL_EO_2.5118864315095824_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2639398 --exp_id FSCL_EO_3.1622776601683795_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5451795 --exp_id FSCL_EO_5.011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4306499 --exp_id FSCL_EO_6.309573444801936_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4052116 --exp_id FSCL_EO_7.943282347242821_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5110479 --exp_id FSCL_EO_0.003981071705534973_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2761015 --exp_id FSCL_EO_0.025118864315095808_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1913283 --exp_id FSCL_EO_10.0_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4652327 --exp_id FSCL_EO_0.0025118864315095794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7893187 --exp_id FSCL_EO_0.03162277660168379_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1220198 --exp_id FSCL_EO_0.039810717055349734_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 508446 --exp_id FSCL_EO_0.005011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3504871 --exp_id FSCL_EO_0.01584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8092381 --exp_id FSCL_EO_3.981071705534973_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8097441 --exp_id FSCL_EO_0.0031622776601683794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6187927 --exp_id FSCL_EO_0.012589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5727330 --exp_id FSCL_EO_0.007943282347242814_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2270638 --exp_id FSCL_EO_0.06309573444801933_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3540761 --exp_id FSCL_EO_0.1_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4724955 --exp_id FSCL_EO_0.12589254117941676_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8954073 --exp_id FSCL_EO_0.0199526231496888_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3087201 --exp_id FSCL_EO_0.001584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2812728 --exp_id FSCL_EO_0.05011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9031424 --exp_id FSCL_EO_1.584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7165483 --exp_id FSCL_EO_1.9952623149688808_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6028306 --exp_id FSCL_EO_0.07943282347242818_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4758756 --exp_id FSCL_EO_0.0012589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8828277 --exp_id FSCL_EO_0.01_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 111379 --exp_id FSCL_EO_0.001995262314968879_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1072313 --exp_id FSCL_EO_0.001_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1043077 --exp_id FSCL_EO_0.00630957344480193_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4642619 --exp_id FSCL_EO_0.15848931924611143_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1773252 --exp_id FSCL_EO_0.19952623149688808_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8206926 --exp_id FSCL_EO_0.25118864315095824_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7560514 --exp_id FSCL_EO_0.31622776601683794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8272651 --exp_id FSCL_EO_0.3981071705534973_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1033678 --exp_id FSCL_EO_0.5011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9138294 --exp_id FSCL_EO_0.6309573444801936_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2989262 --exp_id FSCL_EO_0.7943282347242822_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1019269 --exp_id FSCL_EO_1.0_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8634949 --exp_id FSCL_EO_1.2589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5335029 --exp_id FSCL_EO_2.5118864315095824_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1699681 --exp_id FSCL_EO_3.1622776601683795_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9148487 --exp_id FSCL_EO_5.011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4260637 --exp_id FSCL_EO_6.309573444801936_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9726425 --exp_id FSCL_EO_7.943282347242821_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 342975 --exp_id FSCL_EO_0.003981071705534973_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1686638 --exp_id FSCL_EO_0.025118864315095808_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7047257 --exp_id FSCL_EO_10.0_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5849956 --exp_id FSCL_EO_0.0025118864315095794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2682237 --exp_id FSCL_EO_0.03162277660168379_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7873894 --exp_id FSCL_EO_0.039810717055349734_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2684680 --exp_id FSCL_EO_0.005011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 116107 --exp_id FSCL_EO_0.01584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 523101 --exp_id FSCL_EO_3.981071705534973_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2162035 --exp_id FSCL_EO_0.0031622776601683794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6714796 --exp_id FSCL_EO_0.012589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 302525 --exp_id FSCL_EO_0.007943282347242814_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7383235 --exp_id FSCL_EO_0.06309573444801933_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3924906 --exp_id FSCL_EO_0.1_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3690511 --exp_id FSCL_EO_0.12589254117941676_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4430507 --exp_id FSCL_EO_0.0199526231496888_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9970046 --exp_id FSCL_EO_0.001584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1449008 --exp_id FSCL_EO_0.05011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6203365 --exp_id FSCL_EO_1.584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9939882 --exp_id FSCL_EO_1.9952623149688808_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8456148 --exp_id FSCL_EO_0.07943282347242818_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3591173 --exp_id FSCL_EO_0.0012589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6410818 --exp_id FSCL_EO_0.01_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8181995 --exp_id FSCL_EO_0.001995262314968879_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9506777 --exp_id FSCL_EO_0.001_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6890355 --exp_id FSCL_EO_0.00630957344480193_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4318767 --exp_id FSCL_EO_0.15848931924611143_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6285178 --exp_id FSCL_EO_0.19952623149688808_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5031256 --exp_id FSCL_EO_0.25118864315095824_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2996158 --exp_id FSCL_EO_0.31622776601683794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6980496 --exp_id FSCL_EO_0.3981071705534973_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9414179 --exp_id FSCL_EO_0.5011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3499808 --exp_id FSCL_EO_0.6309573444801936_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5570361 --exp_id FSCL_EO_0.7943282347242822_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8979241 --exp_id FSCL_EO_1.0_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6988124 --exp_id FSCL_EO_1.2589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4870878 --exp_id FSCL_EO_2.5118864315095824_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 925125 --exp_id FSCL_EO_3.1622776601683795_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8019976 --exp_id FSCL_EO_5.011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 209250 --exp_id FSCL_EO_6.309573444801936_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 262360 --exp_id FSCL_EO_7.943282347242821_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 635509 --exp_id FSCL_EO_0.003981071705534973_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7818111 --exp_id FSCL_EO_0.025118864315095808_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9691306 --exp_id FSCL_EO_10.0_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1217389 --exp_id FSCL_EO_0.0025118864315095794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4677097 --exp_id FSCL_EO_0.03162277660168379_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1803122 --exp_id FSCL_EO_0.039810717055349734_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9484755 --exp_id FSCL_EO_0.005011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8259986 --exp_id FSCL_EO_0.01584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7313466 --exp_id FSCL_EO_3.981071705534973_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1342032 --exp_id FSCL_EO_0.0031622776601683794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8577040 --exp_id FSCL_EO_0.012589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7836045 --exp_id FSCL_EO_0.007943282347242814_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2336497 --exp_id FSCL_EO_0.06309573444801933_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2586448 --exp_id FSCL_EO_0.1_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1803497 --exp_id FSCL_EO_0.12589254117941676_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1016881 --exp_id FSCL_EO_0.0199526231496888_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 10951 --exp_id FSCL_EO_0.001584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1943615 --exp_id FSCL_EO_0.05011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1341896 --exp_id FSCL_EO_1.584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6064847 --exp_id FSCL_EO_1.9952623149688808_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6204307 --exp_id FSCL_EO_0.07943282347242818_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5744399 --exp_id FSCL_EO_0.0012589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6457793 --exp_id FSCL_EO_0.01_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9797273 --exp_id FSCL_EO_0.001995262314968879_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4083533 --exp_id FSCL_EO_0.001_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7925387 --exp_id FSCL_EO_0.00630957344480193_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 996043 --exp_id FSCL_EO_0.15848931924611143_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4371244 --exp_id FSCL_EO_0.19952623149688808_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7713201 --exp_id FSCL_EO_0.25118864315095824_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5600145 --exp_id FSCL_EO_0.31622776601683794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1949821 --exp_id FSCL_EO_0.3981071705534973_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9232562 --exp_id FSCL_EO_0.5011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5216573 --exp_id FSCL_EO_0.6309573444801936_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 391665 --exp_id FSCL_EO_0.7943282347242822_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 668139 --exp_id FSCL_EO_1.0_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3098670 --exp_id FSCL_EO_1.2589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9370544 --exp_id FSCL_EO_2.5118864315095824_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5057653 --exp_id FSCL_EO_3.1622776601683795_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6922961 --exp_id FSCL_EO_5.011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1100114 --exp_id FSCL_EO_6.309573444801936_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2148170 --exp_id FSCL_EO_7.943282347242821_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6315463 --exp_id FSCL_EO_0.003981071705534973_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 344079 --exp_id FSCL_EO_0.025118864315095808_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5382096 --exp_id FSCL_EO_10.0_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7854413 --exp_id FSCL_EO_0.0025118864315095794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6349465 --exp_id FSCL_EO_0.03162277660168379_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6029104 --exp_id FSCL_EO_0.039810717055349734_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2327924 --exp_id FSCL_EO_0.005011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 951722 --exp_id FSCL_EO_0.01584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9834875 --exp_id FSCL_EO_3.981071705534973_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9650683 --exp_id FSCL_EO_0.0031622776601683794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3267644 --exp_id FSCL_EO_0.012589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1169490 --exp_id FSCL_EO_0.007943282347242814_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4124644 --exp_id FSCL_EO_0.06309573444801933_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6918873 --exp_id FSCL_EO_0.1_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4780665 --exp_id FSCL_EO_0.12589254117941676_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5035241 --exp_id FSCL_EO_0.0199526231496888_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 603665 --exp_id FSCL_EO_0.001584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7058156 --exp_id FSCL_EO_0.05011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5602699 --exp_id FSCL_EO_1.584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3396625 --exp_id FSCL_EO_1.9952623149688808_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9934469 --exp_id FSCL_EO_0.07943282347242818_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2253619 --exp_id FSCL_EO_0.0012589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4681438 --exp_id FSCL_EO_0.01_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9563107 --exp_id FSCL_EO_0.001995262314968879_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2629564 --exp_id FSCL_EO_0.001_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL -FCLObj EO --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
