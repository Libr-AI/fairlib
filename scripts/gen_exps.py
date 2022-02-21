import itertools as it
import numpy as np

def loguniform(low=0, high=1, size=None):
    return np.power(10, np.random.uniform(low, high, size))

def log_grid(start, stop, number_trails):
    assert stop >= start
    step = (stop-start)/number_trails
    step_index = [i for i in range(number_trails+1)]
    return np.power(10, np.array([start+i*step for i in step_index]))

def grid(start, stop, number_trails):
    assert stop >= start
    step = (stop-start)/number_trails
    step_index = [i for i in range(number_trails+1)]
    return np.array([start+i*step for i in step_index])

exps={}
exps["dataset"]={"Moji", "Bios_gender", "Bios_age", "Bios_both"}
exps["batch_size"]={512,1024,2048}
exps["learning_rate"]=set(log_grid(-4,-2,6))
exps["hidden_size"]={200,300,400}
exps["n_hidden"]={1,2,3}
exps["dropout"]=set(grid(0,0.5,3))
exps["batch_norm"]={True, False}

allNames=exps.keys()
combos = it.product(*(exps[Name] for Name in allNames))

def write_to_batch_files(random_seed=1, repeat=0):
    for id, combo in enumerate(combos):
        with open("scripts/hypertune/vanilla_{}_{}.sh".format(combo[0], repeat),"a+") as f:
            command = "python main.py --project_dir hypertune --dataset {_dataset} --emb_size {_emb_size} --num_classes {_num_classes} --batch_size {_batch_size} --learning_rate {_learning_rate} --hidden_size {_hidden_size} --n_hidden {_n_hidden} --dropout {_dropout}{_batch_norm} --base_seed {_random_seed} --exp_id {_exp_id}"
            # dataset
            _dataset = combo[0]
            _emb_size = 2304 if _dataset == "Moji" else 768
            _num_classes = 2 if _dataset == "Moji" else 28
            _batch_size = combo[1]
            _learning_rate = combo[2]
            _hidden_size = combo[3]
            _n_hidden = combo[4]
            _dropout = combo[5]
            _batch_norm = " --batch_norm" if combo[6] else ""
            _exp_id = "hypertune_vanilla{}".format(id)
            _repeat = repeat
            _random_seed = random_seed
            
            command=command.format(
                _repeat=_repeat,
                _dataset=_dataset,
                _emb_size=_emb_size,
                _num_classes=_num_classes,
                _batch_size=_batch_size,
                _learning_rate=_learning_rate,
                _hidden_size=_hidden_size,
                _n_hidden=_n_hidden,
                _dropout=_dropout,
                _batch_norm=_batch_norm,
                _random_seed=_random_seed,
                _exp_id=_exp_id,
                    )
            
            f.write(command+"\nsleep 2\n")

write_to_batch_files()