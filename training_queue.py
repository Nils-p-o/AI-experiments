from training_loop import run_experiment
import time
# might need to make the params allow dynamic changes in architecture, maybe (like mixing DiffAttn and nGPT, when it works...)

LLaMa = {
    "architecture": "LLaMa",
    "batch_size": 32,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 9,
    "d_ff_mult": 4,
    "groups": 4,
    "dropout": 0.1,
    "lr": 5e-4,
    "warmup_steps": 100,
    "t_0": 5000,
    "t_mult": 1.5,
    "lr_mult": 0.5,
    "seq_len": 128,
    "type": "baseline",
}

DiffTransformer = {
    "architecture": "Diff",
    "batch_size": 32,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 9,
    "d_ff_mult": 4,
    "groups": 4,
    "dropout": 0.1,
    "lr": 5e-4,
    "warmup_steps": 100,
    "t_0": 5000,
    "t_mult": 1.5,
    "lr_mult": 0.5,
    "seq_len": 128,
    "type": "test",
}

experiment_nr = 1
# checking if i fucked up the causal mask
run_experiment(LLaMa)
time.sleep(10)
print(f"Experiment {experiment_nr} completed")
experiment_nr += 1


# setting baseline + testing queue
for i in range(3):
    run_experiment(DiffTransformer)
    print(f"Experiment {experiment_nr} completed")
    experiment_nr += 1

lr_test = DiffTransformer.copy()
lr_test["lr"] = 1e-3

for i in range(2):
    run_experiment(lr_test)
    print(f"Experiment {experiment_nr} completed")
    experiment_nr += 1

mha_test = DiffTransformer.copy()
mha_test["groups"] = 8

for i in range(1):
    run_experiment(mha_test)
    print(f"Experiment {experiment_nr} completed")
    experiment_nr += 1