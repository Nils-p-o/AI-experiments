@echo off
@REM specifically for hypkos computer (for now)
@REM replace paths with C:/Users/nilsp/AppData/Local/Programs/Python/Python311/python.exe c:/Users/nilsp/Documents/AI-experiments/train.py for my computer
@REM # Experiment 1: Baseline LLaMa
C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture LLaMa --d_model 512 --nhead 8 --num_layers 8 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 7e-4 --warmup_steps 500 --t_0 5000 --t_mult 1.5 --lr_mult 0.7 --seq_len 512 --batch_size 32 --t_total 100000 --type baseline

@REM @REM # Experiment 2: Diff transformer
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline

@REM @REM # Experiment 3: Diff transformer with higher lr
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 1e-3 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 1e-3 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test

@REM @REM # Experiment 4: Diff transformer with 8 groups (mha)
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 8 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 8 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test

@REM @REM # Experiment 5: Diff transformer with 7 layers
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 7 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test
@REM C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 7 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test

pause