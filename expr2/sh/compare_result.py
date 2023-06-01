'''
    用于结果对比：
        文件格式：第一列为顶点编号，第二列为顶点的值
'''

import os
import pandas as pd
import sys

if __name__ == "__main__":
    # 运行格式： python3 ./compare_result.py ../out/sssp_result ../out/sssp_result_sum

    path1 = sys.argv[1].strip(' ')
    path2 = sys.argv[2].strip(' ')
    print('参数列表:', str(sys.argv))
    print('path1:', path1)
    print('path2:', path2)

    # path1 = r"././out/pr_delta_pre.txt"
    df1 = pd.read_csv(path1, header=None, sep=' ')
    df1 = df1.sort_values(by=0, ascending=True)  # 按age排列, ascending=False表示降序，为True为升序，默认为True
    df1.to_csv(path1, index=None, columns=None, header=False, sep=' ') # 重新保存文件
    print(df1.head())
    # print(df1.tail())
    print(df1.shape)

    # path2 = r"././out/pr_delta_sum_com.txt"
    df2 = pd.read_csv(path2, header=None, sep=' ')
    df2 = df2.sort_values(by=0, ascending=True)  # 按age排列, ascending=False表示降序，为True为升序，默认为True
    df2.to_csv(path2, index=None, columns=None, header=False, sep=' ')  # 重新保存文件
    print(df2.head())
    # print(df2.tail())
    print(df2.shape)

    print(path1)
    print(path2)
    if(df1.shape != df2.shape):
        print("shapt not same...", df1.shape, df2.shape)

    sp1 = df1.values
    sp2 = df2.values
    wc = 0.0
    sum1 = 0.0
    sum2 = 0.0
    wc_cnt = 0
    # print(sp1[0][1], sp2[0][1], sp2[0][1]/newdf1.shape[0])
    cmp_pair = []
    for i in range(df1.shape[0]):
        a, b = sp1[i][1], sp2[i][1]
        if a < 1e15:
            wc += abs(a - b)
            sum1 += a
            sum2 += b
            if(abs(a - b) > 1e-5):
                wc_cnt += 1
                if len(cmp_pair) < 10:
                    cmp_pair.append(("id=" + str(sp1[i][0]), a, b))
        
    print('nor和sum误差: ', wc)
    print('误差点的个数：', wc_cnt)
    print('cmp_pair:', cmp_pair)
    print('delta_%s sum1 =' % path1, sum1)
    print('delta_%s sum2 =' % path2, sum2)
    print(path1) 
    print(path2)

