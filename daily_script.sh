#!/bin/bash
python data_input.py --prior_day 20210630 
python model.py --mode train
python model.py --mode test
python trade_output.py

