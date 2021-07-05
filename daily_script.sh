#!/bin/bash
python daily_data_input.py --prior_day 20210603 --feature_generate_mode daily 
python daily_model.py --mode train
python daily_model.py --mode test
python daily_trade_output.py

