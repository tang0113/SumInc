#!/usr/bin/env python3
import sys
from subprocess import getstatusoutput
import re
import csv

ingress_prefix = "/mnt/data/nfs/yusong/code/z_Ingress/build"
suminc_prefix = "/mnt/data/nfs/yusong/code/SumInc/build"
serial_prefix_root='/mnt/data/nfs/yusong/ser'
serialization_cmp_prefix='/mnt/data/nfs/yusong/ser/spnode'

def get_dataset():
    dataset = {
                'road_usa': 16651563,
                'europe_osm': 44499591,
                # 'uk-2005': 23311901,
            #    'twitter': 1037947,
                # 'com-friendster'
               }
    return dataset

def run(cmd):
    status, out = getstatusoutput(cmd)
    with open('/tmp/expr2.log', 'a') as fo:
        fo.write(cmd + "\n")
        fo.write(out + "\n")
    if status != 0:
        print("Failed to execute " + cmd)
    return status, out

def run_Ingress():
    dataset = get_dataset()
    serial_prefix = ''
    for app in ['pagerank']: #['pagerank', 'sssp']:
        if app == 'pagerank':
            serial_prefix = serial_prefix_root + '/one'
        elif app == 'sssp':
            serial_prefix = serial_prefix_root + '/two'
        else:
            print('no this type...')
        for name, source in dataset.items(): # 'europe_osm', 'road_usa', 'uk-2005', 'twitter', 'com-friendster'
            percentage=0.01
            directed = ''
            if app not in ['pagerank']:
                directed = '_w'
            path='/mnt/data/nfs/dataset/{name}'.format(name=name)
            efile='{path}/{percentage}/{name}{directed}.base'.format(name=name, path=path, percentage=percentage, directed=directed)
            efile_update='{path}/{percentage}/{name}{directed}.update'.format(name=name, path=path, percentage=percentage, directed=directed)
            efile_updated='{path}/{percentage}/{name}{directed}.updated'.format(name=name, path=path, percentage=percentage, directed=directed)
            vfile='{path}/{name}{directed}.v'.format(name=name, path=path, percentage=percentage, directed=directed)
            workers_num=1
            app_concurrency=16 # 代码中默认值为：-1, -1表示使用全部线程
            portion=0.3
            cilk=1

            print(efile)
            print(efile_update)
            print(vfile)

            # "-out_prefix /tmp/sum_{NAME} "\
            cmd = "mpirun -n {WORKS_NUM} {BIN_PREFIX}/ingress " \
                            "-application {APP} " \
                            "-vfile {VFILE} " \
                            "-efile {EFILE} " \
                            "-efile_update {EFILE_UPDATE} " \
                            "-directed " \
                            "-cilk={CILK} " \
                            "-segmented_partition=false " \
                            "-termcheck_threshold 0.000001 " \
                            "-app_concurrency {APP_CONCURRENCY} " \
                            "-gcn_mr 2 " \
                            "-sssp_source {SOURCE} -php_source {SOURCE} "\
                            "-portion={PORTION} "\
                            "-serialization_prefix {SERIAL_PREFIX} ".format(
                            WORKS_NUM=workers_num,
                            BIN_PREFIX=ingress_prefix,
                            CILK=cilk,
                            APP_CONCURRENCY=app_concurrency,
                            APP=app,
                            VFILE=vfile,
                            EFILE=efile,
                            EFILE_UPDATE=efile_update,
                            NAME=name,
                            SOURCE=source,
                            PORTION=portion,
                            SERIAL_PREFIX=serial_prefix
                            )
            run_times = 1
            print(cmd)
            for curr_round in range(run_times):
                print("Evaluating({ROUND}) {APP} on {NAME}".format(ROUND=curr_round, APP=app, NAME=name))
                status, autoinc_out = run(cmd)
                if status != 0:
                    ok = False
                    print('false.')
                    break

def run_Suminc():
    dataset = get_dataset()
    serial_prefix = ''
    for app in ['pagerank']: #['pagerank', 'sssp']:
        if app == 'pagerank':
            serial_prefix = serial_prefix_root + '/one'
        elif app == 'sssp':
            serial_prefix = serial_prefix_root + '/two'
        else:
            print('no this type...')
        for name, source in dataset.items(): # 'europe_osm', 'road_usa', 'uk-2005', 'twitter', 'com-friendster'
            for max_node_num in [8, 15, 30, 50, 150, 200]: #[8, 15, 30, 50, 150, 200]: # [8, 10, 14, 18, 24, 30, 50, 70, 80, 100, 120, 140, 160]:
                for compress_concurrency in [-1]: # [-1, 4]: #[-1, 8]: # 并行压缩线程数量, -1表示不用多线程
                    for compress in [1]: #[0, 1]: # 是否使用压缩图
                        percentage=0.01
                        # path='/home/neu/dataset/{name}'.format(name=name)
                        directed = ''
                        if app not in ('pagerank'):
                            directed = '_w'
                        path='/mnt/data/nfs/dataset/{name}'.format(name=name)
                        efile='{path}/{percentage}/{name}{directed}.base'.format(name=name, path=path, percentage=percentage, directed=directed)
                        efile_update='{path}/{percentage}/{name}{directed}.update'.format(name=name, path=path, percentage=percentage, directed=directed)
                        efile_updated='{path}/{percentage}/{name}{directed}.updated'.format(name=name, path=path, percentage=percentage, directed=directed)
                        vfile='{path}/{name}{directed}.v'.format(name=name, path=path, percentage=percentage, directed=directed)
                        workers_num=1
                        app_concurrency=16 # 代码中默认值为：-1, -1表示使用全部线程
                        portion=0.3
                        min_node_num=5
                        build_index_concurrency=32 # 计算索引时的线程数量
                        cilk=1

                        print(efile)
                        print(efile_update)
                        print(vfile)

                        # inc
                        # cmd = "mpirun -n $work_num ./ingress -application pagerank -efile ${efile} -efile_update ${efile_update} -vfile ${vfile}  -out_prefix /tmp/sum_${name} -app_concurrency ${app_concurrency} -termcheck_threshold ${termcheck_threshold} -compress=1 -portion=${portion} -min_node_num=${min_node_num} -max_node_num=${max_node_num}".format(efile=efile, efile_update=efile_update, vfile=vfile, name=name, app_concurrency=app_concurrency, termcheck_threshold=termcheck_threshold, )

                        # "-out_prefix /tmp/sum_{NAME} "\
                        cmd = "mpirun -n {WORKS_NUM} {BIN_PREFIX}/ingress " \
                                        "-application {APP} " \
                                        "-vfile {VFILE} " \
                                        "-efile {EFILE} " \
                                        "-efile_update {EFILE_UPDATE} " \
                                        "-directed " \
                                        "-cilk={CILK} " \
                                        "-segmented_partition=false " \
                                        "-termcheck_threshold 0.000001 " \
                                        "-app_concurrency {APP_CONCURRENCY} " \
                                        "-gcn_mr 2 " \
                                        "-sssp_source {SOURCE} -php_source {SOURCE} "\
                                        "-compress={COMPRESS} "\
                                        "-portion={PORTION} "\
                                        "-min_node_num={MIN_NODE_NUM} "\
                                        "-max_node_num={MAX_NODE_NUM} "\
                                        "-serialization_prefix {SERIAL_PREFIX} "\
                                        "-serialization_cmp_prefix {SERIAL_CMP_PREFIX} "\
                                        "-build_index_concurrency {BUILD_INDEX_CONCURRENCY} "\
                                        "-compress_concurrency={COMPRES_CONCURRENCY} ".format(
                                        WORKS_NUM=workers_num,
                                        BIN_PREFIX=suminc_prefix,
                                        CILK=cilk,
                                        APP_CONCURRENCY=app_concurrency,
                                        APP=app,
                                        VFILE=vfile,
                                        EFILE=efile,
                                        EFILE_UPDATE=efile_update,
                                        COMPRESS=compress,
                                        NAME=name,
                                        WORKERS_NUM=workers_num,
                                        SOURCE=source,
                                        PORTION=portion,
                                        MIN_NODE_NUM=min_node_num,
                                        MAX_NODE_NUM=max_node_num,
                                        SERIAL_PREFIX=serial_prefix,
                                        SERIAL_CMP_PREFIX=serialization_cmp_prefix,
                                        BUILD_INDEX_CONCURRENCY=build_index_concurrency,
                                        COMPRES_CONCURRENCY=compress_concurrency)
                        run_times = 1
                        print(cmd)
                        for curr_round in range(run_times):
                            print("Evaluating({ROUND}) {APP} on {NAME}".format(ROUND=curr_round, APP=app, NAME=name))
                            status, autoinc_out = run(cmd)
                            if status != 0:
                                ok = False
                                print('false.')
                                break

if __name__ == '__main__':
    # python3 run.py ingress
    if sys.argv[1] == 'ingress':
        run_Ingress()
    elif sys.argv[1] == 'suminc':
        run_Suminc()
    else:
        sys.exit("Invalid arg: " + sys.argv[1])