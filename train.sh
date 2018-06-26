#!/bin/bash

seed=12345

python3 train.py \
--output_dir=model-train-data-${seed} \
--n_epochs=20 \
--batch_size=16 \
--embed_size=300 \
--num_classes=8 \
--lstm_dropout=0.3 \
--attn_dropout=0.3 \
--lr=0.0002 \
--lstm_dim=256 \
--fc_dim=128 \
--use_cuda \
--seed=${seed}
