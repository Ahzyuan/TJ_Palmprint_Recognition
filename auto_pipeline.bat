@echo off
cls

conda activate my_env

cd Dataset
rar x ROI.rar ROI

python dataset_generate.py -o Dataset/ROI -d Dataset

python train.py

pause
exit