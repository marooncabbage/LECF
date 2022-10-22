#!/bin/bash
for ((i=1;i<100;i+=2))
do
  learning_rate=$( echo "$i*0.0001"|bc )
  python run_t1.py --dataset yelp --cuda 0 --patience 10 --lr $learning_rate
done