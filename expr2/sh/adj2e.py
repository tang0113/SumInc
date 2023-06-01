#
#   adj文件转e文件
#
#   eq: python3 adj2e.py /mnt/data/nfs/yusong/dataset/large/com-friendster/com-friendster.all.cmty.txt

#!/usr/bin/env python3
import os
import sys
import random
import time

def main():
    path = sys.argv[1]
    prefix = os.path.dirname(path)
    filename = os.path.basename(path)
    base = filename
    ext = ''

    if '.' in filename:
        base = os.path.splitext(filename)[0]
        if filename.split('.')[-1] == 'c': # scanpp file
            exit(0)
        ext = '.'.join(os.path.splitext(filename)[1:])

    print(prefix)
    print(base)
    print(ext)

    savepath = prefix + r"/" + base + '.e'
    print(savepath)

    split_ = '\t' # 每个的分割符
    num_lines = 0
    header = False
    with open(savepath, 'w') as fo:
        with open(path, 'r') as fi:
            for line in fi:
                line = line.strip()

                if len(line) == 0 or line.startswith('#') or line.startswith('%') or line == '\n':
                    continue
                num_lines += 1
                if header and num_lines == 1:
                    print('header:', line)
                    continue
                parts = line.split(split_)
                if len(parts) > 1:
                    for i in range(1, len(parts)):
                        fo.write('%s %s\n' % (parts[0], parts[i]))
    print('write finish...')

if __name__ == '__main__':
    time_start=time.time()
    main()
    time_end=time.time()
    print('time cost',time_end-time_start)