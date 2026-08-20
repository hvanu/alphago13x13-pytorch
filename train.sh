#!/bin/bash

set -e

python3 main.py --sgf-dir sgf --pretrain-epochs 10 --selfplay-iters 0 --batch-size 64

