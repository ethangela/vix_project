# VIX future forecast project source code

This repository contains codes for the project 'VIX Futures Prediction'. 


---
### Main requirements
1. Python 3.5+
2. PyTorch 1.9.0
3. Other essential dependencies like numpy, sklearn, pandas, and math, etc.


By typing the command

./daily_script.sh 

into Linux Bash shell, the daily prediction auto-generation tool will start, run, terminate and present the results automatically.



---
### Usage
You might find that within the shell script daily_script.sh, there are two important configuration hyperparameters worth updating manually:

--prior_day, type=str, e.g., 20210603

--feature_generate_mode, type=str, e.g., all/daily






