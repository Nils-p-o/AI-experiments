#!/bin/bash

# Experiment 1: Baseline LLaMa
python train.py --architecture LLaMa --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline

# Experiment 2: Diff transformer
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type baseline

# Experiment 3: Diff transformer with higher lr
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 1e-3 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 1e-3 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test

# Experiment 4: Diff transformer with 8 groups (mha)
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 8 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 9 --d_ff_mult 4 --groups 8 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test

# Experiment 5: Diff transformer with 7 layers
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 7 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test
python train.py --architecture Diff --d_model 512 --nhead 8 --num_layers 7 --d_ff_mult 4 --groups 4 --dropout 0.1 --lr 5e-4 --warmup_steps 100 --t_0 5000 --t_mult 1.5 --lr_mult 0.5 --seq_len 128 --batch_size 32 --type test