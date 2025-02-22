@echo off

C:/Users/estu0/AppData/Local/Programs/Python/Python310/python.exe c:/Users/estu0/OneDrive/Documents/VScode_code/AI-experiments/inference.py --d_model 512 --nhead 8 --num_layers 6 --d_ff_mult 4 --groups 8 --seq_len 128 --dropout 0.1 --model_path "models/DINT_wikitext2.pth" --max_length 350 --prompt "derivatives are" --temperature 1.0



pause