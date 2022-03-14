#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_gender_FSCL

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Bios_gender_FSCL_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Bios_gender_FSCL_%J.out

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

python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6100289 --exp_id FSCL_0.00630957344480193_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6249445 --exp_id FSCL_0.00039810717055349735_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00039810717055349735 --fcl_lambda_g 0.00039810717055349735
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1775635 --exp_id FSCL_0.0026607250597988083_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0026607250597988083 --fcl_lambda_g 0.0026607250597988083
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 271763 --exp_id FSCL_0.03548133892335753_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03548133892335753 --fcl_lambda_g 0.03548133892335753
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9838791 --exp_id FSCL_0.0707945784384138_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0707945784384138 --fcl_lambda_g 0.0707945784384138
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8058052 --exp_id FSCL_0.08413951416451948_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.08413951416451948 --fcl_lambda_g 0.08413951416451948
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6353534 --exp_id FSCL_0.00019952623149688788_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00019952623149688788 --fcl_lambda_g 0.00019952623149688788
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3521343 --exp_id FSCL_0.0005623413251903491_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0005623413251903491 --fcl_lambda_g 0.0005623413251903491
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 313824 --exp_id FSCL_0.0018836490894898002_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0018836490894898002 --fcl_lambda_g 0.0018836490894898002
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3395218 --exp_id FSCL_0.0001_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001 --fcl_lambda_g 0.0001
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 965445 --exp_id FSCL_0.0004731512589614803_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0004731512589614803 --fcl_lambda_g 0.0004731512589614803
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5452180 --exp_id FSCL_0.029853826189179603_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.029853826189179603 --fcl_lambda_g 0.029853826189179603
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 515429 --exp_id FSCL_0.001333521432163324_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001333521432163324 --fcl_lambda_g 0.001333521432163324
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7114901 --exp_id FSCL_0.0022387211385683377_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0022387211385683377 --fcl_lambda_g 0.0022387211385683377
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4736842 --exp_id FSCL_0.003758374042884443_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003758374042884443 --fcl_lambda_g 0.003758374042884443
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7273618 --exp_id FSCL_0.0009440608762859235_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0009440608762859235 --fcl_lambda_g 0.0009440608762859235
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6401139 --exp_id FSCL_0.0006683439175686149_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0006683439175686149 --fcl_lambda_g 0.0006683439175686149
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8234151 --exp_id FSCL_0.0031622776601683794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2710092 --exp_id FSCL_0.012589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2407543 --exp_id FSCL_0.014962356560944329_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.014962356560944329 --fcl_lambda_g 0.014962356560944329
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5462112 --exp_id FSCL_0.1_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 810427 --exp_id FSCL_0.0007943282347242813_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0007943282347242813 --fcl_lambda_g 0.0007943282347242813
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 254880 --exp_id FSCL_0.001122018454301963_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001122018454301963 --fcl_lambda_g 0.001122018454301963
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3786662 --exp_id FSCL_0.00023713737056616554_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00023713737056616554 --fcl_lambda_g 0.00023713737056616554
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5467505 --exp_id FSCL_0.007498942093324558_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007498942093324558 --fcl_lambda_g 0.007498942093324558
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6507661 --exp_id FSCL_0.001584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 451006 --exp_id FSCL_0.042169650342858224_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.042169650342858224 --fcl_lambda_g 0.042169650342858224
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1882500 --exp_id FSCL_0.0044668359215096305_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0044668359215096305 --fcl_lambda_g 0.0044668359215096305
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5978590 --exp_id FSCL_0.008912509381337459_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.008912509381337459 --fcl_lambda_g 0.008912509381337459
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6871215 --exp_id FSCL_0.00033496543915782756_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00033496543915782756 --fcl_lambda_g 0.00033496543915782756
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3997724 --exp_id FSCL_0.00011885022274370189_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00011885022274370189 --fcl_lambda_g 0.00011885022274370189
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 181182 --exp_id FSCL_0.01778279410038923_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01778279410038923 --fcl_lambda_g 0.01778279410038923
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8698656 --exp_id FSCL_0.021134890398366454_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.021134890398366454 --fcl_lambda_g 0.021134890398366454
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5050398 --exp_id FSCL_0.0501187233627272_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0501187233627272 --fcl_lambda_g 0.0501187233627272
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6783510 --exp_id FSCL_0.059566214352901034_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.059566214352901034 --fcl_lambda_g 0.059566214352901034
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9094137 --exp_id FSCL_0.005308844442309879_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005308844442309879 --fcl_lambda_g 0.005308844442309879
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9310327 --exp_id FSCL_0.010592537251772886_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.010592537251772886 --fcl_lambda_g 0.010592537251772886
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8620127 --exp_id FSCL_0.025118864315095794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095794 --fcl_lambda_g 0.025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6655040 --exp_id FSCL_0.00016788040181225607_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00016788040181225607 --fcl_lambda_g 0.00016788040181225607
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5619798 --exp_id FSCL_0.0001412537544622754_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001412537544622754 --fcl_lambda_g 0.0001412537544622754
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2704718 --exp_id FSCL_0.0002818382931264455_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0002818382931264455 --fcl_lambda_g 0.0002818382931264455
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1901812 --exp_id FSCL_0.00630957344480193_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7878765 --exp_id FSCL_0.00039810717055349735_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00039810717055349735 --fcl_lambda_g 0.00039810717055349735
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1345519 --exp_id FSCL_0.0026607250597988083_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0026607250597988083 --fcl_lambda_g 0.0026607250597988083
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4339665 --exp_id FSCL_0.03548133892335753_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03548133892335753 --fcl_lambda_g 0.03548133892335753
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3380710 --exp_id FSCL_0.0707945784384138_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0707945784384138 --fcl_lambda_g 0.0707945784384138
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9959447 --exp_id FSCL_0.08413951416451948_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.08413951416451948 --fcl_lambda_g 0.08413951416451948
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4783453 --exp_id FSCL_0.00019952623149688788_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00019952623149688788 --fcl_lambda_g 0.00019952623149688788
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 73039 --exp_id FSCL_0.0005623413251903491_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0005623413251903491 --fcl_lambda_g 0.0005623413251903491
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8773790 --exp_id FSCL_0.0018836490894898002_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0018836490894898002 --fcl_lambda_g 0.0018836490894898002
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8502590 --exp_id FSCL_0.0001_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001 --fcl_lambda_g 0.0001
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9948400 --exp_id FSCL_0.0004731512589614803_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0004731512589614803 --fcl_lambda_g 0.0004731512589614803
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8374544 --exp_id FSCL_0.029853826189179603_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.029853826189179603 --fcl_lambda_g 0.029853826189179603
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 411735 --exp_id FSCL_0.001333521432163324_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001333521432163324 --fcl_lambda_g 0.001333521432163324
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 132896 --exp_id FSCL_0.0022387211385683377_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0022387211385683377 --fcl_lambda_g 0.0022387211385683377
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5289343 --exp_id FSCL_0.003758374042884443_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003758374042884443 --fcl_lambda_g 0.003758374042884443
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6208798 --exp_id FSCL_0.0009440608762859235_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0009440608762859235 --fcl_lambda_g 0.0009440608762859235
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4887009 --exp_id FSCL_0.0006683439175686149_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0006683439175686149 --fcl_lambda_g 0.0006683439175686149
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1975834 --exp_id FSCL_0.0031622776601683794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6690953 --exp_id FSCL_0.012589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4544828 --exp_id FSCL_0.014962356560944329_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.014962356560944329 --fcl_lambda_g 0.014962356560944329
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 615986 --exp_id FSCL_0.1_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6864440 --exp_id FSCL_0.0007943282347242813_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0007943282347242813 --fcl_lambda_g 0.0007943282347242813
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3026321 --exp_id FSCL_0.001122018454301963_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001122018454301963 --fcl_lambda_g 0.001122018454301963
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 70697 --exp_id FSCL_0.00023713737056616554_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00023713737056616554 --fcl_lambda_g 0.00023713737056616554
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9272545 --exp_id FSCL_0.007498942093324558_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007498942093324558 --fcl_lambda_g 0.007498942093324558
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1015050 --exp_id FSCL_0.001584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5059564 --exp_id FSCL_0.042169650342858224_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.042169650342858224 --fcl_lambda_g 0.042169650342858224
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2130391 --exp_id FSCL_0.0044668359215096305_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0044668359215096305 --fcl_lambda_g 0.0044668359215096305
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7744412 --exp_id FSCL_0.008912509381337459_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.008912509381337459 --fcl_lambda_g 0.008912509381337459
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1585425 --exp_id FSCL_0.00033496543915782756_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00033496543915782756 --fcl_lambda_g 0.00033496543915782756
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9158802 --exp_id FSCL_0.00011885022274370189_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00011885022274370189 --fcl_lambda_g 0.00011885022274370189
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3305265 --exp_id FSCL_0.01778279410038923_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01778279410038923 --fcl_lambda_g 0.01778279410038923
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1749837 --exp_id FSCL_0.021134890398366454_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.021134890398366454 --fcl_lambda_g 0.021134890398366454
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4177226 --exp_id FSCL_0.0501187233627272_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0501187233627272 --fcl_lambda_g 0.0501187233627272
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3647316 --exp_id FSCL_0.059566214352901034_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.059566214352901034 --fcl_lambda_g 0.059566214352901034
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6584073 --exp_id FSCL_0.005308844442309879_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005308844442309879 --fcl_lambda_g 0.005308844442309879
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 797672 --exp_id FSCL_0.010592537251772886_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.010592537251772886 --fcl_lambda_g 0.010592537251772886
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8484940 --exp_id FSCL_0.025118864315095794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095794 --fcl_lambda_g 0.025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 993433 --exp_id FSCL_0.00016788040181225607_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00016788040181225607 --fcl_lambda_g 0.00016788040181225607
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9116315 --exp_id FSCL_0.0001412537544622754_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001412537544622754 --fcl_lambda_g 0.0001412537544622754
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2519863 --exp_id FSCL_0.0002818382931264455_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0002818382931264455 --fcl_lambda_g 0.0002818382931264455
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5701531 --exp_id FSCL_0.00630957344480193_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3183072 --exp_id FSCL_0.00039810717055349735_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00039810717055349735 --fcl_lambda_g 0.00039810717055349735
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4328718 --exp_id FSCL_0.0026607250597988083_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0026607250597988083 --fcl_lambda_g 0.0026607250597988083
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 498106 --exp_id FSCL_0.03548133892335753_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03548133892335753 --fcl_lambda_g 0.03548133892335753
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7675782 --exp_id FSCL_0.0707945784384138_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0707945784384138 --fcl_lambda_g 0.0707945784384138
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5612530 --exp_id FSCL_0.08413951416451948_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.08413951416451948 --fcl_lambda_g 0.08413951416451948
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3785845 --exp_id FSCL_0.00019952623149688788_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00019952623149688788 --fcl_lambda_g 0.00019952623149688788
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3781851 --exp_id FSCL_0.0005623413251903491_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0005623413251903491 --fcl_lambda_g 0.0005623413251903491
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8206934 --exp_id FSCL_0.0018836490894898002_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0018836490894898002 --fcl_lambda_g 0.0018836490894898002
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1497362 --exp_id FSCL_0.0001_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001 --fcl_lambda_g 0.0001
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 416053 --exp_id FSCL_0.0004731512589614803_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0004731512589614803 --fcl_lambda_g 0.0004731512589614803
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2231145 --exp_id FSCL_0.029853826189179603_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.029853826189179603 --fcl_lambda_g 0.029853826189179603
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6725202 --exp_id FSCL_0.001333521432163324_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001333521432163324 --fcl_lambda_g 0.001333521432163324
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2685038 --exp_id FSCL_0.0022387211385683377_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0022387211385683377 --fcl_lambda_g 0.0022387211385683377
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4914787 --exp_id FSCL_0.003758374042884443_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003758374042884443 --fcl_lambda_g 0.003758374042884443
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9207721 --exp_id FSCL_0.0009440608762859235_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0009440608762859235 --fcl_lambda_g 0.0009440608762859235
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7263960 --exp_id FSCL_0.0006683439175686149_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0006683439175686149 --fcl_lambda_g 0.0006683439175686149
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 670818 --exp_id FSCL_0.0031622776601683794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3454772 --exp_id FSCL_0.012589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7460919 --exp_id FSCL_0.014962356560944329_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.014962356560944329 --fcl_lambda_g 0.014962356560944329
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3372168 --exp_id FSCL_0.1_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 48640 --exp_id FSCL_0.0007943282347242813_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0007943282347242813 --fcl_lambda_g 0.0007943282347242813
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8183419 --exp_id FSCL_0.001122018454301963_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001122018454301963 --fcl_lambda_g 0.001122018454301963
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4759525 --exp_id FSCL_0.00023713737056616554_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00023713737056616554 --fcl_lambda_g 0.00023713737056616554
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6245055 --exp_id FSCL_0.007498942093324558_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007498942093324558 --fcl_lambda_g 0.007498942093324558
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1606930 --exp_id FSCL_0.001584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 318284 --exp_id FSCL_0.042169650342858224_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.042169650342858224 --fcl_lambda_g 0.042169650342858224
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6998653 --exp_id FSCL_0.0044668359215096305_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0044668359215096305 --fcl_lambda_g 0.0044668359215096305
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8330249 --exp_id FSCL_0.008912509381337459_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.008912509381337459 --fcl_lambda_g 0.008912509381337459
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 374119 --exp_id FSCL_0.00033496543915782756_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00033496543915782756 --fcl_lambda_g 0.00033496543915782756
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7756846 --exp_id FSCL_0.00011885022274370189_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00011885022274370189 --fcl_lambda_g 0.00011885022274370189
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6532130 --exp_id FSCL_0.01778279410038923_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01778279410038923 --fcl_lambda_g 0.01778279410038923
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3663591 --exp_id FSCL_0.021134890398366454_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.021134890398366454 --fcl_lambda_g 0.021134890398366454
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7815111 --exp_id FSCL_0.0501187233627272_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0501187233627272 --fcl_lambda_g 0.0501187233627272
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3783476 --exp_id FSCL_0.059566214352901034_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.059566214352901034 --fcl_lambda_g 0.059566214352901034
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7751101 --exp_id FSCL_0.005308844442309879_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005308844442309879 --fcl_lambda_g 0.005308844442309879
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 477552 --exp_id FSCL_0.010592537251772886_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.010592537251772886 --fcl_lambda_g 0.010592537251772886
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8700281 --exp_id FSCL_0.025118864315095794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095794 --fcl_lambda_g 0.025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6704768 --exp_id FSCL_0.00016788040181225607_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00016788040181225607 --fcl_lambda_g 0.00016788040181225607
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6324863 --exp_id FSCL_0.0001412537544622754_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001412537544622754 --fcl_lambda_g 0.0001412537544622754
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3089053 --exp_id FSCL_0.0002818382931264455_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0002818382931264455 --fcl_lambda_g 0.0002818382931264455
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4868748 --exp_id FSCL_0.00630957344480193_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9118286 --exp_id FSCL_0.00039810717055349735_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00039810717055349735 --fcl_lambda_g 0.00039810717055349735
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6271794 --exp_id FSCL_0.0026607250597988083_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0026607250597988083 --fcl_lambda_g 0.0026607250597988083
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3748082 --exp_id FSCL_0.03548133892335753_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03548133892335753 --fcl_lambda_g 0.03548133892335753
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6178952 --exp_id FSCL_0.0707945784384138_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0707945784384138 --fcl_lambda_g 0.0707945784384138
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9956729 --exp_id FSCL_0.08413951416451948_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.08413951416451948 --fcl_lambda_g 0.08413951416451948
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3219403 --exp_id FSCL_0.00019952623149688788_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00019952623149688788 --fcl_lambda_g 0.00019952623149688788
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8289757 --exp_id FSCL_0.0005623413251903491_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0005623413251903491 --fcl_lambda_g 0.0005623413251903491
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3875712 --exp_id FSCL_0.0018836490894898002_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0018836490894898002 --fcl_lambda_g 0.0018836490894898002
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7398514 --exp_id FSCL_0.0001_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001 --fcl_lambda_g 0.0001
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9611248 --exp_id FSCL_0.0004731512589614803_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0004731512589614803 --fcl_lambda_g 0.0004731512589614803
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1070636 --exp_id FSCL_0.029853826189179603_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.029853826189179603 --fcl_lambda_g 0.029853826189179603
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9056671 --exp_id FSCL_0.001333521432163324_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001333521432163324 --fcl_lambda_g 0.001333521432163324
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1365279 --exp_id FSCL_0.0022387211385683377_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0022387211385683377 --fcl_lambda_g 0.0022387211385683377
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6380978 --exp_id FSCL_0.003758374042884443_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003758374042884443 --fcl_lambda_g 0.003758374042884443
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2059803 --exp_id FSCL_0.0009440608762859235_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0009440608762859235 --fcl_lambda_g 0.0009440608762859235
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3593631 --exp_id FSCL_0.0006683439175686149_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0006683439175686149 --fcl_lambda_g 0.0006683439175686149
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1097887 --exp_id FSCL_0.0031622776601683794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7223515 --exp_id FSCL_0.012589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3648090 --exp_id FSCL_0.014962356560944329_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.014962356560944329 --fcl_lambda_g 0.014962356560944329
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6960129 --exp_id FSCL_0.1_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1813636 --exp_id FSCL_0.0007943282347242813_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0007943282347242813 --fcl_lambda_g 0.0007943282347242813
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7871004 --exp_id FSCL_0.001122018454301963_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001122018454301963 --fcl_lambda_g 0.001122018454301963
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3906966 --exp_id FSCL_0.00023713737056616554_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00023713737056616554 --fcl_lambda_g 0.00023713737056616554
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6537148 --exp_id FSCL_0.007498942093324558_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007498942093324558 --fcl_lambda_g 0.007498942093324558
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9083548 --exp_id FSCL_0.001584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 843893 --exp_id FSCL_0.042169650342858224_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.042169650342858224 --fcl_lambda_g 0.042169650342858224
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4084864 --exp_id FSCL_0.0044668359215096305_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0044668359215096305 --fcl_lambda_g 0.0044668359215096305
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8780394 --exp_id FSCL_0.008912509381337459_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.008912509381337459 --fcl_lambda_g 0.008912509381337459
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3170990 --exp_id FSCL_0.00033496543915782756_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00033496543915782756 --fcl_lambda_g 0.00033496543915782756
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9345431 --exp_id FSCL_0.00011885022274370189_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00011885022274370189 --fcl_lambda_g 0.00011885022274370189
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9074107 --exp_id FSCL_0.01778279410038923_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01778279410038923 --fcl_lambda_g 0.01778279410038923
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6715913 --exp_id FSCL_0.021134890398366454_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.021134890398366454 --fcl_lambda_g 0.021134890398366454
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1145208 --exp_id FSCL_0.0501187233627272_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0501187233627272 --fcl_lambda_g 0.0501187233627272
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8809000 --exp_id FSCL_0.059566214352901034_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.059566214352901034 --fcl_lambda_g 0.059566214352901034
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4290179 --exp_id FSCL_0.005308844442309879_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005308844442309879 --fcl_lambda_g 0.005308844442309879
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9122351 --exp_id FSCL_0.010592537251772886_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.010592537251772886 --fcl_lambda_g 0.010592537251772886
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9241477 --exp_id FSCL_0.025118864315095794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095794 --fcl_lambda_g 0.025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6057017 --exp_id FSCL_0.00016788040181225607_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00016788040181225607 --fcl_lambda_g 0.00016788040181225607
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1881902 --exp_id FSCL_0.0001412537544622754_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001412537544622754 --fcl_lambda_g 0.0001412537544622754
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9324727 --exp_id FSCL_0.0002818382931264455_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0002818382931264455 --fcl_lambda_g 0.0002818382931264455
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9900342 --exp_id FSCL_0.00630957344480193_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7508406 --exp_id FSCL_0.00039810717055349735_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00039810717055349735 --fcl_lambda_g 0.00039810717055349735
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6935544 --exp_id FSCL_0.0026607250597988083_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0026607250597988083 --fcl_lambda_g 0.0026607250597988083
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2660983 --exp_id FSCL_0.03548133892335753_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03548133892335753 --fcl_lambda_g 0.03548133892335753
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8321025 --exp_id FSCL_0.0707945784384138_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0707945784384138 --fcl_lambda_g 0.0707945784384138
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7763095 --exp_id FSCL_0.08413951416451948_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.08413951416451948 --fcl_lambda_g 0.08413951416451948
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4582277 --exp_id FSCL_0.00019952623149688788_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00019952623149688788 --fcl_lambda_g 0.00019952623149688788
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6533102 --exp_id FSCL_0.0005623413251903491_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0005623413251903491 --fcl_lambda_g 0.0005623413251903491
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9500212 --exp_id FSCL_0.0018836490894898002_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0018836490894898002 --fcl_lambda_g 0.0018836490894898002
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 575137 --exp_id FSCL_0.0001_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001 --fcl_lambda_g 0.0001
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7985996 --exp_id FSCL_0.0004731512589614803_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0004731512589614803 --fcl_lambda_g 0.0004731512589614803
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5826010 --exp_id FSCL_0.029853826189179603_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.029853826189179603 --fcl_lambda_g 0.029853826189179603
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3606525 --exp_id FSCL_0.001333521432163324_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001333521432163324 --fcl_lambda_g 0.001333521432163324
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8548029 --exp_id FSCL_0.0022387211385683377_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0022387211385683377 --fcl_lambda_g 0.0022387211385683377
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7874007 --exp_id FSCL_0.003758374042884443_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003758374042884443 --fcl_lambda_g 0.003758374042884443
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3865760 --exp_id FSCL_0.0009440608762859235_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0009440608762859235 --fcl_lambda_g 0.0009440608762859235
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6722954 --exp_id FSCL_0.0006683439175686149_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0006683439175686149 --fcl_lambda_g 0.0006683439175686149
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9446340 --exp_id FSCL_0.0031622776601683794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8237140 --exp_id FSCL_0.012589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6952430 --exp_id FSCL_0.014962356560944329_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.014962356560944329 --fcl_lambda_g 0.014962356560944329
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8419566 --exp_id FSCL_0.1_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3208391 --exp_id FSCL_0.0007943282347242813_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0007943282347242813 --fcl_lambda_g 0.0007943282347242813
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7863102 --exp_id FSCL_0.001122018454301963_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001122018454301963 --fcl_lambda_g 0.001122018454301963
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7470987 --exp_id FSCL_0.00023713737056616554_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00023713737056616554 --fcl_lambda_g 0.00023713737056616554
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9319892 --exp_id FSCL_0.007498942093324558_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007498942093324558 --fcl_lambda_g 0.007498942093324558
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 238173 --exp_id FSCL_0.001584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3978630 --exp_id FSCL_0.042169650342858224_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.042169650342858224 --fcl_lambda_g 0.042169650342858224
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9209582 --exp_id FSCL_0.0044668359215096305_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0044668359215096305 --fcl_lambda_g 0.0044668359215096305
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2918186 --exp_id FSCL_0.008912509381337459_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.008912509381337459 --fcl_lambda_g 0.008912509381337459
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2911440 --exp_id FSCL_0.00033496543915782756_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00033496543915782756 --fcl_lambda_g 0.00033496543915782756
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9484169 --exp_id FSCL_0.00011885022274370189_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00011885022274370189 --fcl_lambda_g 0.00011885022274370189
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9010126 --exp_id FSCL_0.01778279410038923_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01778279410038923 --fcl_lambda_g 0.01778279410038923
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 144033 --exp_id FSCL_0.021134890398366454_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.021134890398366454 --fcl_lambda_g 0.021134890398366454
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8128234 --exp_id FSCL_0.0501187233627272_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0501187233627272 --fcl_lambda_g 0.0501187233627272
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4376950 --exp_id FSCL_0.059566214352901034_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.059566214352901034 --fcl_lambda_g 0.059566214352901034
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8427330 --exp_id FSCL_0.005308844442309879_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005308844442309879 --fcl_lambda_g 0.005308844442309879
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9211226 --exp_id FSCL_0.010592537251772886_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.010592537251772886 --fcl_lambda_g 0.010592537251772886
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7465776 --exp_id FSCL_0.025118864315095794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095794 --fcl_lambda_g 0.025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6768805 --exp_id FSCL_0.00016788040181225607_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00016788040181225607 --fcl_lambda_g 0.00016788040181225607
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8238515 --exp_id FSCL_0.0001412537544622754_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0001412537544622754 --fcl_lambda_g 0.0001412537544622754
sleep 2
python main.py --project_dir FairSCL --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7831073 --exp_id FSCL_0.0002818382931264455_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0002818382931264455 --fcl_lambda_g 0.0002818382931264455
sleep 2
