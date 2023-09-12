# VIX futures forecast project source code

This repository contains codes for the project 'VIX Futures Prediction'. 


---
### Main requirements
1. Python 3.5+
2. PyTorch 1.9.0
3. Other essential dependencies like numpy, sklearn, pandas, and math, etc.


---
### Usage
1. By typing the command ```./daily_script.sh ``` in Linux Shell, the daily prediction auto-generation tool will automatically start, run, terminate, and present the results.
2. In case of data not being maintained, discontinuity, or when the program has not been run for a long time, you can update the following two parameters ```daily_script.sh``` manually to allow the model to calibrate itself automatically:
     - ```prior_day```, type: str, example: ```'20210603'```.
     - ```feature_generate_mode```, type: str, example: ```'all/daily'```.








