@echo off
cls

: activate conda env, specify your anaconda path
set CONDA_PATH=E:\Softwaredata\Anaconda\Scripts 
call "%CONDA_PATH%\activate.bat" tj_palm

: prepare data
mkdir Assets\ROI
rar x Assets\ROI.rar Assets\ROI
python dataset_generate.py -o Assets\ROI -d Dataset

: train
python train.py

: test
: python test.py -c Config/hyp_TJ.json -w Results/saved_model/best_<year>-<month>-<day>T<hour>-<minute>.pth

cmd /k