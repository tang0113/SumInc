# To build:
# make

# The input parameters of these applications are as follows:
# ./toolkits/pagerank [path] [vertices] [iterations]
# ./toolkits/cc [path] [vertices]
# ./toolkits/sssp [path] [vertices] [root]
# ./toolkits/bfs [path] [vertices] [root]
# ./toolkits/bc [path] [vertices] [root]

# Note: [vertices] equal max_id+1 in edgefile.

root_path=/mnt2/neu/yusong
app_root_path=/mnt2/neu/yusong
app=sssp
dataname=google
# last_fix=_w
last_fix=_w_ud
efilepath=${root_path}/dataset/large/${dataname}/0.0001/${dataname}${last_fix}.updated
echo $efilepath

#####################################
# sssp [file] [vertices] [root] [threads_num]
g++ ${app_root_path}/code/gemini-graph/utils/GraphToBinary.cpp -o GraphToBinary && ${app_root_path}/code/gemini-graph/GraphToBinary ${efilepath} -w=1
${app_root_path}/code/gemini-graph/toolkits/sssp ${efilepath}.b 875713   3 16

#####################################
# php [file] [vertices] [root] [iterations] [threads_num]
# g++ ${app_root_path}/gemini-graph//utils/ConvertWeight.cpp -o ConvertWeight && ./ConvertWeight ${efilepath} -w=1 -type=gemini_php
# ./toolkits/php ${efilepath}.cb 39454746   3311901 100 8

#####################################
# ppr [file] [vertices] [root] [iterations] [threads_num]
# g++ ${app_root_path}/code/gemini-graph/utils/GraphToBinary.cpp -o GraphToBinary && ${app_root_path}/code/gemini-graph/utils/GraphToBinary ${efilepath} -w=0
# ${app_root_path}/code/gemini-graph/toolkits/ppr ${efilepath}.b 32 0 100 8


#####################################
# pagerank [file] [vertices] [iterations]
# g++ ./utils/GraphToBinary.cpp -o GraphToBinary && ./GraphToBinary ${efilepath} -w=0
# ./toolkits/pagerank ${efilepath}.b 18484117 74



#------------------------------------
# compare reslut
# python3 /mnt/data/nfs/yusong/code/SumInc/expr2/sh/compare_result.py /mnt/data/nfs/yusong/result/sum_pagerank/result_frag_0 /mnt/data/nfs/yusong/code/gemini-graph/result

# python3 /mnt/data/nfs/yusong/code/SumInc/expr2/sh/compare_result.py /mnt/data/nfs/yusong/code/libgrape-lite/build/output_pr_grape/result_frag_0 /mnt/data/nfs/yusong/result/sum_pagerank/result_frag_0



#------------------------------------
# file:       [max_id+1]  source
# uk-2002:    18484117 198319
# uk-2005:    39454746   3311901
# europe_osm: 50912018   10000
# road_usa:   23947347 0
# arabic-2005:  22743892 0
# it-2004:    41290682 0
# gsh-2015-tpd: 30809122 6792
# sk-2005: 50636073 0
# webbase-2001: 115657290 1000

# (uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001)
# max_odegree_id=(198319 23311901 10000 0 0 0 6792 0 1000) # modified
# node_size=(18484117 39454746 50912018 23947347 22743892 41290682 30809122 50636073 115657290) # modified