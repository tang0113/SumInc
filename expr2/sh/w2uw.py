#
# 将带权重的图转为无权图
#

#!/usr/bin/env python3
import os
import sys
import random
import time

def w2ud():
    # create vm
    path = '/mnt/data/nfs/dataset/uk-2005/0.03/uk-2005_w.base'
    print("Creating VM")
    num_lines = 0
    split_ = ' '
    # uw_path = path.replace('_w', '')
    uw_path = '/mnt/data/nfs/dataset/uk-2005/0.03/uk-2005.base'
    fp_unweight = open(uw_path, 'w')
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()

            if len(line) == 0 or line.startswith('#') or line.startswith('%') or line == '\n':
                continue
            num_lines += 1
            parts = line.split(split_)
            u = int(parts[0])
            v = int(parts[1])
            fp_unweight.write('%d %d\n' % (u, v))
    fp_unweight.close()


if __name__ == '__main__':
    time_start=time.time()
    w2ud()
    time_end=time.time()
    print('time cost',time_end-time_start)