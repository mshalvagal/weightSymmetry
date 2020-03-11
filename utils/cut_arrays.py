import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Net2WiderNet')
    parser.add_argument('--teacher-dir', type=str, 
        help='location of teacher training curves to cut')

    args = parser.parse_args()

    for f_name in os.listdir(args.teacher_dir):
        if f_name.endswith('.npy'):
            x = np.load(os.path.join(args.teacher_dir, f_name))
            np.save(os.path.join(args.teacher_dir, f_name), x[:len(x)//2])

if __name__ == '__main__':
    main()
