#
#   顶点重新编号
#
#!/usr/bin/env python3
import os
import sys
import random
import time

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python newId.py path [-w=0/1]")
    
    # cmd: python3 newId.py /mnt/data/nfs/dataset/road_usa/0.01/road_usa.base

    print(sys.argv)

    path = sys.argv[1]
    have_weight = False
    split_ = ' '
    if len(sys.argv) >= 3:
        if sys.argv[2] == '-w=0':
            have_weight = False
        elif sys.argv[2] == '-w=1':
            have_weight = True
        else:
            sys.exit("Illegal arg: " + sys.argv[3])

    prefix = os.path.dirname(path)
    filename = os.path.basename(path)
    base = filename
    ext = ''
    random.seed(10)

    if '.' in filename:
        base = os.path.splitext(filename)[0]
        ext = '.'.join(os.path.splitext(filename)[1:])

    num_lines = 0
    vm = {}
    vertex_num = 0

    # create vm
    print("Creating VM")
    ids = set()
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()

            if len(line) == 0 or line.startswith('#') or line.startswith('%') or line == '\n':
                continue
            num_lines += 1
            parts = line.split(split_)
            if len(parts):
                print(line)
                break
            u = int(parts[0])
            v = int(parts[1])

            ids.add(u)
            ids.add(v)

            if u not in vm:
                vm[u] = vertex_num
                vertex_num += 1
            if v not in vm:
                vm[v] = vertex_num
                vertex_num += 1
    
    print("max=", max(ids))
    print("size=", len(ids))
    # Load graph
    # print("Loading graph")
    # num_lines = 0
    # with open(path, 'r') as fi:
    #     for line in fi:
    #         line = line.strip()

    #         if len(line) == 0 or line.startswith('#') or line.startswith('%'):
    #             continue
    #         num_lines += 1
    #         parts = line.split(split_)
    #         u_gid = vm[int(parts[0])]
    #         v_gid = vm[int(parts[1])]
    #         # skip self cycle
    #         if u_gid == v_gid:
    #             continue
    #         if u_gid not in G:
    #             G[u_gid] = dict()
    #         oes = G[u_gid]

    #         if have_weight:
    #             weight = random.randint(1, 64)
    #             oes[v_gid] = weight
    #         else:
    #             oes[v_gid] = 1

if __name__ == '__main__':
    time_start=time.time()
    main()
    time_end=time.time()
    print('time cost',time_end-time_start)