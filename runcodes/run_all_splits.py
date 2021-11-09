import argparse
import subprocess
import os
from subprocess import Popen
import sys


if __name__ == '__main__':

    arguments = sys.argv
    cuda_devices = arguments[1].split(",")
    splits_to_run = arguments[2].split(",")

    for i, ele in enumerate(splits_to_run):
        command = ["python"]
        command.extend(arguments[3:])
        command.extend(['--split', ele])
        command.extend(['--cudad', cuda_devices[i]])
        print(" ".join(command))
        output_file = open(f"logs/run_all_split{ele}.log", 'w+')
        error_file = open(f"logs/run_all_split{ele}.err", 'w+')
        print("Output file name = ", output_file)
        print("Error file name = ", error_file)
        Popen(command, stdout=output_file, stderr=error_file)

