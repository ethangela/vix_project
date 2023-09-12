# VIX futures forecast project source code

---
### Introduction
1. This repository contains the codes for the project 'VIX Futures Prediction'.
2. This code was previously private. After removing the pre-trained parameters and training details, it has been made public for learning reference.
3. The author is Ethan Yang Sun.


---
### Main requirements
1. Python 3.5+
2. PyTorch 1.9.0
3. Other essential dependencies like numpy, sklearn, pandas, and math, etc.


---
### Usage
1. By typing the command ```sh ./daily_script.sh ``` in Linux Shell, the daily prediction auto-generation tool will automatically start, run, terminate, and present the results.
2. In case of data not being maintained, discontinuity, or when the program has not been run for a long time, you can manually update the following two arguments directly after the command  ```sh daily_script.sh``` to allow the model to calibrate itself automatically:
     - ```-- prior_day```, e.g., ```20210603```.
     - ```-- feature_generate_mode```, e.g., ```all``` or ```daily```.








