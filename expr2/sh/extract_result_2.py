#-*- coding: UTF-8 -*-   
'''
    用于解析运行结果: 通过result.txt文件提取本次运行结果，然后通过读取上次保存的csv
    文件，将内容合并，并生成excel和csv文件
    针对规范内容提取, 例如,下面的语句提取a和b 
    eq:  xxxx #a: b

    cmd: python3 extract_result_2.py log_tes
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
                # if col > 0:
                    # old_col = col 
                    # time_one = 0
                    # for index in one_iter:
                    #     if not pd.isnull(df.loc[index, old_col]):
                    #         try:
                    #             time_one += float(df.loc[index, old_col])
                    #         except:
                    #             pass
                    # time_two = 0
                    # for index in two_iter:
                    #     if not pd.isnull(df.loc[index, old_col]):
                    #         try:
                    #             time_two += float(df.loc[index, old_col])
                    #         except:
                    #             pass
                    # df.loc['time_one', old_col] = time_one
                    # df.loc['time_two', old_col] = time_two
                col += 1
                df.loc['cmd', col] = line
                find = True
                match = re.match("^.*?/large/(.*?)/.*$", line)
                if match:
                    df.loc['dataset', col] = match.groups()[0]
                match = re.match("^.*?/(0.*?)/.*$", line)
                if match:
                    df.loc['update_rate', col] = match.groups()[0]
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
                match = re.match("^.*?-compress_type=(.*?) -.*$", line)
                if match:
                    df.loc['compress_type', col] = match.groups()[0]
                # print(line)
                continue
            match = re.match("^.*?Thread num: (.*?)$", line)
            if match:
                df.loc['Thread num', col] = int(match.groups()[0])

            match = re.match("^.*?Mem: (.*?) MB$", line)
            if match:
                df.loc['Mem', col] = int(match.groups()[0])

            match = re.match("^.*?#(.*?): (.*?)$", line)
            if match:
                df.loc[match.groups()[0], col] = match.groups()[1]

        # if col > 0:
        #     old_col = col 
        #     time_one = 0
        #     for index in one_iter:
        #         if not pd.isnull(df.loc[index, old_col]):
        #             try:
        #                 time_one += float(df.loc[index, old_col])
        #             except:
        #                 pass
        #     time_two = 0
        #     for index in two_iter:
        #         if not pd.isnull(df.loc[index, old_col]):
        #             try:
        #                 time_two += float(df.loc[index, old_col])
        #             except:
        #                 pass
        #     df.loc['time_one', old_col] = time_one
        #     df.loc['time_two', old_col] = time_two

    # savepath = path[0:path.rfind('.')]
    savepath = path
    df.to_excel(savepath + '.xlsx')
    df.to_csv(savepath + '.csv')
    print('save to', savepath+'.xlsx')

if __name__ == '__main__':
    path = sys.argv[1]
    print("get data frome", path)
    run(path)