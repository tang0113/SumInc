# mtx文件转边文件
# 边文件,每行只包含源顶点和目的顶点
#
# cmd: python3 mtx2e.py /mnt/data/nfs/yusong/dataset/large/friendster/friendster.mtx -header
# cmd: python3 mtx2e.py /mnt/data/nfs/yusong/dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx -header
import os
import sys
import time

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python mtx2e.py [path] [-header]")
    split_ = ' ' # 行分割符

    path = sys.argv[1]
    header = False
    if len(sys.argv) == 3:
        if sys.argv[2] != '-header':
            sys.exit("Illegal arg: " + sys.argv[2])
        else:
            header = True

    prefix = os.path.dirname(path)
    filename = os.path.basename(path)
    base = filename
    ext = ''

    if '.' in filename:
        base = os.path.splitext(filename)[0]
        ext = '.e'

    save_path = "{PREFIX}/{NAME}{EXT}".format(PREFIX=prefix, NAME=base, EXT=ext)
    print('save_path=', save_path)

    num_lines = 0
    with open(save_path, 'w') as fo:
        with open(path, 'r') as fi:
            for line in fi:
                line = line.strip()

                if len(line) == 0 or line.startswith('#') or line.startswith('%'):
                    continue
                num_lines += 1
                if header and num_lines == 1:
                    print('head:', line)
                    continue
                parts = line.split(split_)
                u = int(parts[0])
                v = int(parts[1])
                weight = None
                if len(parts) == 3:
                    fo.write('%d %d %s\n' % (u, v, parts[2]))
                else:
                    fo.write('%d %d\n' % (u, v))
    print('finish!')

if __name__ == '__main__':
    time_start=time.time()
    main()
    time_end=time.time()
    print('time cost',time_end-time_start)