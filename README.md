# VIX future forcast project source code

This repository provides codes for the project 'VIX Future Prediction'. By typing the command

./daily_script.sh 

into Linux Bash shell, the daily prediction auto-generation tool will start, run, terminate and present the results automatically.



## Requirements
PyTorch 1.9.0 and Python 3.5+. Other dependecies include numpy, sklearn, pandas and math etc.



## Usage
You might find that within the shell script daily_script.sh, there are two important configuration hyperparameters worth updating manually:

--prior_day, type=str, e.g., 20210603

--feature_generate_mode, type=str, e.g., all/daily




