#!/bin/bash
##usage: ./train_forecast.sh <forecast_end_date(m/d/y)>

regions=("jaipur" "delhi" "mumbai" "bengaluru urban" "bengaluru rural" "ahmedabad" "pune")
region_types=("district" "state" "district" "district" "district" "district" "district")

rlen=${#regions[@]}

for(( i=0; i<${rlen}; i++ ));
        do python train_eval_plot.py --region "${regions[$i]}" --region_type "${region_types[$i]}" --forecast_end_date "$1";
done