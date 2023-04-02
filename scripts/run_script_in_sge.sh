#!/bin/bash

## Job name
#$ -N train-ser-classifier-with-whisper-4l

## Working directory
#$ -wd $HOME/speechemotionrecognition

## STDOUT/STDERR output files
#$ -o $HOME/speechemotionrecognition/logs
#$ -e $HOME/speechemotionrecognition/logs

## Devices
#$ -l gpu=1

source $HOME/miniconda3/bin/activate speechemotionrecognition
python $HOME/speechemotionrecognition/scripts/train-classifier.py
