#!/bin/bash
for ((i=1;i<100;i+=2))
do
  learning_rate=$( echo "$i*0.0001"|bc )
  #echo $learning_rate
  python run_t1.py --dataset Amazon-CD --cuda 0 --patience 20 --lr $learning_rate
done
