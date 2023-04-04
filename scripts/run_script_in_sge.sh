#!/bin/bash

## GPU queue
#$ -q fox.q

## Job name
#$ -N train-ser-classifier-with-whisper-4l

## Working directory
#$ -wd $HOME/speechemotionrecognition

## 2&>1
#$ -j y

## STDOUT/STDERR output file
#$ -o $HOME/speechemotionrecognition/logs

## Devices
#$ -l gpu=1

source $HOME/miniconda3/bin/activate speechemotionrecognition
python $HOME/speechemotionrecognition/scripts/train-classifier.py
