#!/bin/bash
for ((i=0;i<100;i++))
do
  python run_t1.py --dataset Amazon-CD --cuda 1
  python run_t1.py --dataset Amazon-Book --cuda 1
  python run_t1.py --dataset yelp --cuda 1
done