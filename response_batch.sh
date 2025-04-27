#!/bin/bash

# Name your submission so you can recognize it in squeue. Our example is "my-test":
#SBATCH --job-name=my-test
# Send the job output to a file of your choosing. Our example is "my_results.txt":
#SBATCH --output=my_response_results.txt
# Request GPUs:
#SBATCH --gres=gpu:1
# Request CPUs:
#SBATCH --ntasks=4
# Request 1GiB of RAM per CPU:
#SBATCH --mem-per-cpu=1024

hostname
python3 full_response_generation.py
