#
#   生成增量图数据集
#
#   删边/增边/为没有出度的点加边(入邻居中随机选一个)
#   对于无向图在增删边时，需要对入边和出边同时删除
# 

#!/usr/bin/env python3
import os
import sys
import random
import time


def pop_first(d: dict):
    for k in d:
        return k, d.pop(k)


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python gen.py [path] [percentage] [-w=0/1] [-header]") # -w=0无权， -w=1有权
    
    # cmd: python3 gen_inc2.py /mnt/data/nfs/yusong/dataset/large/google/google.e 0.01 -w=0

    path = sys.argv[1]
    percentage = float(sys.argv[2])
    header = False
    have_weight = False
    directed = True
    split_ = ' '
    if len(sys.argv) >= 4:
        if sys.argv[3] == '-w=0':
            have_weight = False
        elif sys.argv[3] == '-w=1':
            have_weight = True
        else:
            sys.exit("Illegal arg: " + sys.argv[3])
    if len(sys.argv) == 5:
        if sys.argv[4] != '-header':
            sys.exit("Illegal arg: " + sys.argv[3])
        else:
            header = True

    prefix = os.path.dirname(path)
    filename = os.path.basename(path)
    base = filename
    ext = ''
    random.seed(10)

    if '.' in filename:
        base = os.path.splitext(filename)[0]
        ext = '.'.join(os.path.splitext(filename)[1:])
    if filename.find("_ud") != -1:
        directed = False

    num_lines = 0
    vm = {}
    vertex_num = 0

    # create vm
    print("Creating VM")
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()

            if len(line) == 0 or line.startswith('#') or line.startswith('%') or line == '\n':
                continue
            num_lines += 1
            if header and num_lines == 1:  # 
                print('header:', line)
                continue
            parts = line.split(split_)
            u = int(parts[0])
            v = int(parts[1])

            if u not in vm:
                vm[u] = vertex_num
                vertex_num += 1
            if v not in vm:
                vm[v] = vertex_num
                vertex_num += 1

    add_size = int(num_lines * percentage / 2)
    del_size = int(num_lines * percentage / 2)

    print("Vertex num: ", vertex_num)
    print("Add size: ", add_size)
    print("Del size: ", del_size)

    if not os.path.exists('%s/%.4f' % (prefix, percentage)):
        os.mkdir('%s/%.4f' % (prefix, percentage))
    if have_weight:
        base += '_w'
    fp_vm = open('%s/%s_%.4f.v' % (prefix, base, percentage), 'w')
    fp_base = open('%s/%.4f/%s.base' % (prefix, percentage, base), 'w')
    fp_append = open('%s/%.4f/%s.update' % (prefix, percentage, base), 'w')
    fp_updated = open('%s/%.4f/%s.updated' % (prefix, percentage, base), 'w')

    print('base edge:', '%s/%.4f/%s.base' % (prefix, percentage, base))

    G = {}
    G_in = {} # in degree

    # Load graph
    print("Loading graph")
    num_lines = 0
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()

            if len(line) == 0 or line.startswith('#') or line.startswith('%'):
                continue
            num_lines += 1
            if header and num_lines == 1:
                continue
            parts = line.split(split_)
            u_gid = vm[int(parts[0])]
            v_gid = vm[int(parts[1])]
            # skip self cycle
            if u_gid == v_gid:
                continue
            if u_gid not in G:
                G[u_gid] = dict()
            oes = G[u_gid]

            if v_gid not in G_in:
                G_in[v_gid] = dict()
            ies = G_in[v_gid]

            if have_weight:
                weight = random.randint(1, 64)
                oes[v_gid] = weight
                ies[u_gid] = weight
            else:
                oes[v_gid] = 1
                ies[u_gid] = 1

    print("Compensating degree")
    # Compensate 0-degree
    cnt = 0
    for u_gid in range(vertex_num):
        if u_gid not in G:
            cnt += 1
            G[u_gid] = dict()
            oes = G[u_gid]
            if u_gid not in G_in:
                G_in[u_gid] = dict()
            ies = G_in[u_gid]

            v_gid = u_gid
            while v_gid == u_gid:
                if len(ies) == 0:
                    v_gid = random.randrange(0, vertex_num)
                else:
                    iadj = list(ies.keys())
                    v_gid = iadj[random.randrange(0, len(iadj))]
            if have_weight:
                weight = random.randint(1, 64)
                oes[v_gid] = weight
            else:
                oes[v_gid] = 1
    print("0-degree_num=", cnt)
    print("Generating add")
    processed_edges = set()
    for u_gid in G:
        oes_u = G[u_gid]
        # u_gid -> v_gid
        if directed == False:
            for v_gid in oes_u:
                if len(oes_u) > 1 and add_size > 2:
                    oes_v = G[v_gid]
                    if len(oes_v) > 1 and add_size > 2:


                    return k, d.pop(k)
        if directed == True:
            while len(oes) > 1 and add_size > 0:
                v_gid, weight = pop_first(oes)
                if have_weight:
                    fp_append.write('a %d %d %s\n' % (u_gid, v_gid, weight))
                    fp_updated.write('%d %d %s\n' % (u_gid, v_gid, weight))
                else:
                    fp_append.write('a %d %d\n' % (u_gid, v_gid))
                    fp_updated.write('%d %d\n' % (u_gid, v_gid))
                processed_edges.add((u_gid, v_gid))
                add_size -= 1
        # for v_gid in oes:
        #     weight = oes[v_gid]
        #     fp_base.write('%d %d %s\n' % (u_gid, v_gid, weight))
        #     fp_updated.write('%d %d %s\n' % (u_gid, v_gid, weight))
        #     visited_edges.add((u_gid, v_gid))

    print("Generating del")
    for u_gid in G:
        oes = G[u_gid]
        while len(oes) > 2 and del_size > 0:
            v_gid, weight = pop_first(oes)
            if have_weight:
                fp_append.write('d %d %d %s\n' % (u_gid, v_gid, weight))
                fp_base.write('%d %d %s\n' % (u_gid, v_gid, weight))
            else:
                fp_append.write('d %d %d\n' % (u_gid, v_gid))
                fp_base.write('%d %d\n' % (u_gid, v_gid))
            processed_edges.add((u_gid, v_gid))
            del_size -= 1

    for u_gid in G:
        oes = G[u_gid]
        for v_gid in oes:
            if have_weight:
                weight = oes[v_gid]
                fp_base.write('%d %d %s\n' % (u_gid, v_gid, weight))
                fp_updated.write('%d %d %s\n' % (u_gid, v_gid, weight))
            else:
                fp_base.write('%d %d\n' % (u_gid, v_gid))
                fp_updated.write('%d %d\n' % (u_gid, v_gid))


    print("Writing vm")
    for v in range(vertex_num):
        fp_vm.write('%d\n' % v)

    fp_vm.close()
    fp_base.close()
    fp_append.close()
    fp_updated.close()


if __name__ == '__main__':
    time_start=time.time()
    main()
    time_end=time.time()
    print('time cost',time_end-time_start)
