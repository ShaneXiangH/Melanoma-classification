# Group 3
胡圣然 11711011  
吴江宁 11710217  
喻家骏 11711102  
朱以辰 11711512  
向昕昊 11712617

# Structure
- checkpoint (trained models)
  - cnn.pth
  - densenet.pth
  - effnet-b0.pth
  - effnet-b0-2.pth
  - googlenet.pth
- models (model classes)
  - init.py
  - cnn.py
  - densenet.py
  - efficientnet.py
  - googlenet.py
- main.py (train api)
- process_data.py (data preprocessing)
- README.md
- requirements.txt
- test.py (test api)
- utils.py

# How to run
Log in 10.20.107.171 by group3 account:  
Username: group3  
Password: cs324group3  

We suggest you use the base conda environment (conda location: /home/group3/anaconda3/bin/conda)  

Or otherwise:  

conda create --name xxenv --file requirements.txt  

NOTE: some packages might not exist in default channels, so you might have to add channels manually or use 

anaconda search -t conda XXpackage_name to install.

## Train
python main.py  
parameters(optional):  
--lr (float, learning rate, default=0.01)

## Test
python test.py   
parameters(optional):  
--data_root (str, root path of dataset, default='/home/group3/DataSet')  
--csv (str, file name of csv, default='test_set.csv')  
--img_folder (str, directory name of images, default='Test_set')  
--target (str, 'test' for no target or otherwise for dataset with target, default='test')  
--model (str, 'single' to use the best single model and 'ensemble' to use the ensemble model, default='ensemble')

### Note: 
1. Just to use default parameters is OK
2. For the ensemble model, it takes a long time to run

