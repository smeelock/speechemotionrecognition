#!/bin/bash

## GPU queue
# -q gpustd.q

## Job name
#$ -N train-ser-classifier-with-whisper-4l-gpu

## Working directory
#$ -wd $HOME/speechemotionrecognition

## STDOUT/STDERR output files
#$ -o $HOME/speechemotionrecognition/logs
#$ -e $HOME/speechemotionrecognition/logs

## Devices
#$ -hard -l gpu=1

source $HOME/miniconda3/bin/activate speechemotionrecognition
python $HOME/speechemotionrecognition/scripts/train-classifier.py
