#-*- coding: UTF-8 -*-   
'''
    用于解析运行结果: 通过result.txt文件提取本次运行结果，然后通过读取上次保存的csv
    文件，将内容合并，并生成excel和csv文件
    
'''
import re
import pandas as pd
import sys

def run(path):
    df = pd.DataFrame()
    col = 0
    row = 0
    one_iter = ['calculate index', 'pre compute', 'run algorithm', 'correct deviation']
    two_iter = ['inc compress', 'inc calculate index', 'inc pre compute', 'inc algorithm', 'inc correct deviation']

    with open(path, 'r') as fi:
        find = False
        row += 1

        for line in fi:
            line = line.strip()
            # print(line)
            # match = re.match("^.*?Batch time: (.*?) sec$", line)
            match = re.match("^mpirun -n", line)
            # print(match)
            if match:
                if col > 0:
                    old_col = col 
                    time_one = 0
                    for index in one_iter:
                        if not pd.isnull(df.loc[index, old_col]):
                            try:
                                time_one += float(df.loc[index, old_col])
                            except:
                                pass
                    time_two = 0
                    for index in two_iter:
                        if not pd.isnull(df.loc[index, old_col]):
                            try:
                                time_two += float(df.loc[index, old_col])
                            except:
                                pass
                    df.loc['time_one', old_col] = time_one
                    df.loc['time_two', old_col] = time_two
                col += 1
                df.loc['cmd', col] = line
                find = True
                match = re.match("^.*?/dataset/(.*?)/.*$", line)
                if match:
                    df.loc['dataset', col] = match.groups()[0]
                match = re.match("^.*?-application (.*?) -.*$", line)
                if match:
                    df.loc['app', col] = match.groups()[0]
                match = re.match("^.*?-compress_concurrency=((\-|\+?)[0-9]+).*$", line)
                if match:
                    df.loc['compress_concurrency', col] = match.groups()[0]
                else:
                    df.loc['compress_concurrency', col] = 1
                match = re.match("^.*?-build_index_concurrency (.*?) -.*$", line)
                if match:
                    df.loc['build_index_concurrency', col] = int(match.groups()[0])
                match = re.match("^.*?-cilk=(.*?) -.*$", line)
                if match:
                    df.loc['cilk', col] = match.groups()[0]
                # print(line)
                continue
            # df.loc['a', col] = 0
            # df.loc['b', col] = 0
            match = re.match("^.*?Thread num: (.*?)$", line)
            if match:
                df.loc['Thread num', col] = int(match.groups()[0])

            match = re.match("^.*?Mem: (.*?) MB$", line)
            if match:
                df.loc['Mem', col] = int(match.groups()[0])

            match = re.match("^.*?graph edges_num: (.*?) nodes_num:(.*?)$", line)
            if match:
                df.loc['edges_num', col] = int(match.groups()[0])
                df.loc['nodes_num', col] = int(match.groups()[1])

            match = re.match("^.*?global_spn_num=(.*?)$", line)
            if match:
                df.loc['global_spn_num', col] = int(match.groups()[0])
            
            match = re.match("^.*?global_spn_com_num=(.*?)$", line)
            if match:
                df.loc['global_spn_com_num', col] = int(match.groups()[0])
            
            match = re.match("^.*?global_inner_edges_num=(.*?)$", line)
            if match:
                df.loc['global_inner_edges_num', col] = int(match.groups()[0])

            match = re.match("^.*?global_bound_edges_num=(.*?)$", line)
            if match:
                df.loc['global_bound_edges_num', col] = int(match.groups()[0])

            match = re.match("^.*?global_spn_com_num/nodes_num=(.*?)$", line)
            if match:
                df.loc['global_spn_com_num/nodes_num', col] = float(match.groups()[0])

            match = re.match("^.*?supernodes_index/edges_num=(.*?)$", line)
            if match:
                df.loc['supernodes_index/edges_num', col] = float(match.groups()[0])
            
            match = re.match("^.*?max_node_num=(.*?)$", line)
            if match:
                df.loc['max_node_num', col] = int(match.groups()[0])

            match = re.match("^.*?max_inner_edges_num=(.*?)$", line)
            if match:
                df.loc['max_inner_edges_num', col] = int(match.groups()[0])
            
            match = re.match("^.*?max_bound_edges_num=(.*?)$", line)
            if match:
                df.loc['max_bound_edges_num', col] = int(match.groups()[0])
            
            match = re.match("^.*?MAX_NODE_NUM=(.*?)$", line)
            if match:
                df.loc['MAX_NODE_NUM', col] = int(match.groups()[0])

            match = re.match("^.*?MIN_NODE_NUM=(.*?)$", line)
            if match:
                df.loc['MIN_NODE_NUM', col] = int(match.groups()[0])

            # match = re.match("^.*? (sum_[0-9])=([0-9]+)$", line)
            # if match:
            #     df.loc['f'+match.groups()[0], col] = int(match.groups()[1])
            
            match = re.match("^.*?iter step: (.*?)$", line)
            if match:
                try:
                    if pd.notnull(df.loc['step1', col]):
                        df.loc['step2', col] = int(match.groups()[0])
                    else:
                        df.loc['step1', col] = int(match.groups()[0])
                except:
                    df.loc['step1', col] = int(match.groups()[0]) 
            
            match = re.match("^.*?find supernode: (.*?) sec$", line)
            if match:
                df.loc['find supernode', col] = float(match.groups()[0])
            
            match = re.match("^.*?- calculate index: (.*?) sec$", line)
            if match:
                df.loc['calculate index', col] = float(match.groups()[0])
            
            match = re.match("^.*?- pre compute: (.*?) sec$", line)
            if match:
                df.loc['pre compute', col] = float(match.groups()[0])
            
            match = re.match("^.*?run algorithm: (.*?) sec$", line)
            if match:
                df.loc['run algorithm', col] = float(match.groups()[0])
            
            match = re.match("^.*?correct deviation: (.*?) sec$", line)
            if match:
                try:
                    if pd.notnull(df.loc['correct deviation', col]):
                        df.loc['inc correct deviation', col] = float(match.groups()[0])
                    else:
                        df.loc['correct deviation', col] = float(match.groups()[0])
                except:
                    df.loc['correct deviation', col] = float(match.groups()[0])
            
            match = re.match("^.*?reloadGraph: (.*?) sec$", line)
            if match:
                df.loc['reloadGraph', col] = float(match.groups()[0])

            # inc rate
            match = re.match("^.*?inccalculate_spnode_ids.size=([0-9]+)?.*$", line)
            if match:
                df.loc['inccalculate_spnode_ids.size', col] = int(match.groups()[0])

            match = re.match("^.*?recalculate_spnode_ids.size=([0-9]+)?.*$", line)
            if match:
                df.loc['recalculate_spnode_ids.size', col] = int(match.groups()[0])

            match = re.match("^.*?reset_edges.size=([0-9]+)?.*$", line)
            if match:
                df.loc['reset_edges.size', col] = int(match.groups()[0])

            match = re.match("^.*?%=(.*?)$", line)
            if match:
                df.loc['cpr((inccal+recal)/allNodeNum)', col] = float(match.groups()[0])
            
            # inc compute
            match = re.match("^.*?inc compress: (.*?) sec$", line)
            if match:
                df.loc['inc compress', col] = float(match.groups()[0])

            match = re.match("^.*?inc calculate index: (.*?) sec$", line)
            if match:
                df.loc['inc calculate index', col] = float(match.groups()[0])
            
            match = re.match("^.*?inc pre compute: (.*?) sec$", line)
            if match:
                df.loc['inc pre compute', col] = float(match.groups()[0])

            match = re.match("^.*?inc algorithm: (.*?) sec$", line)
            if match:
                df.loc['inc algorithm', col] = float(match.groups()[0])

            match = re.match("^.*?Check failed:.*?$", line)
            if match:
                print('error[line %d]:' % row, line)
                df.loc['error', col] = 'WA'

        if col > 0:
            old_col = col 
            time_one = 0
            for index in one_iter:
                if not pd.isnull(df.loc[index, old_col]):
                    try:
                        time_one += float(df.loc[index, old_col])
                    except:
                        pass
            time_two = 0
            for index in two_iter:
                if not pd.isnull(df.loc[index, old_col]):
                    try:
                        time_two += float(df.loc[index, old_col])
                    except:
                        pass
            df.loc['time_one', old_col] = time_one
            df.loc['time_two', old_col] = time_two

    savepath = './expr2.log'[0:'./expr2.log'.rindex('.')]
    df.to_excel(savepath + '.xlsx')
    df.to_csv(savepath + '.csv')

if __name__ == '__main__':
    path = './expr2.log'
    run(path)