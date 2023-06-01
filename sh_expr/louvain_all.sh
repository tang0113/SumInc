
echo -e "\n Executing louvain.sh...\n"
# ipath=../dataset/graph
# opath=../dataset/graph

# python3 /mnt/data/nfs/yusong/code/SumInc/expr2/sh/gen_inc2.py /mnt/data/nfs/yusong/dataset/large/soc-twitter/soc-twitter.e 0.0001 -w=0 

root_path=/mnt2/neu/yusong
for name in uk-2002 uk-2005 arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001 #soc-livejournal
do
    # name: {uk-2002, europe_osm, road_usa}
    percentage=0.0001 # 0.3000 0.4000
    max_comm_size=5000 # it-2004:10000,gsh-2015-tpd:30000,sk-2005:100000,others: 5000
    max_level=2 # web-graph: 2, road-graph: 4
    fix= #_ud #.random
    beta=0.80
    # inc
    ipath=${root_path}/dataset/expr_data_10/${name}/${percentage}/${name}${fix}.base
    opath=${root_path}/dataset/louvain_bin/expr_data_10_${name}_${percentage}${fix}
    # no-inc
    # ipath=${root_path}/dataset/large/${name}/${name}.e
    # opath=${root_path}/dataset/louvain_bin/${name}

    echo ipath=${ipath}
    echo opath=${opath}

    louvain_path=${root_path}/code/louvain/gen-louvain

    # 格式转换
    ${louvain_path}/convert -i ${ipath} -o ${opath}.bin
    echo "----------------------1---------------"

    # 聚类
    cmd="${louvain_path}/louvain ${opath}.bin -l -1 -v -q 0 -e 0.001 -a ${max_level} -m ${max_comm_size} > ${opath}.tree_${max_comm_size}"  # q=0: modularity, q=10: suminc
    echo $cmd
    eval $cmd
    echo "----------------------2---------------"

    # exit

    # 显示树状结构信息（层级数级别和每个级别的节点）
    ${louvain_path}/hierarchy ${opath}.tree_${max_comm_size}
    # # 显示给定级别的节点对社区的归属哪个树
    level=`expr ${max_level} - 1`
    echo level=$level
    ${louvain_path}/hierarchy ${opath}.tree_${max_comm_size} -l ${level} > ${opath}_node2comm_level_${max_comm_size}
    echo "----------------------3--------------"

    #--------------------------------------------------------------------------------------------
    # 为SumInc提供超点数据
    echo ${louvain_path}/getSpNode.cc
    g++ ${louvain_path}/tools/getSpNode.cc -o ${louvain_path}/tools/getSpNode
    cmd="${louvain_path}/tools/getSpNode ${opath}_node2comm_level_${max_comm_size} ${ipath} ${max_comm_size}"
    echo $cmd
    eval $cmd
    echo "----------------------4---------------"

    #---------------------------------------------------------------------------------------------
    # ./matrix ${path}.tree -l ${level} > ${path}_X_level${level}
    cmd="mpirun -n 1 ${root_path}/code/SumInc/build/ingress -application pagerank -vfile /home/yusong/dataset/${name}/${name}.v -efile ${ipath} -directed=1 -cilk=true -termcheck_threshold 1 -app_concurrency 16 -compress=1 -portion=1 -min_node_num=2 -max_node_num=${max_comm_size} -sssp_source=0 -compress_concurrency=1 -build_index_concurrency=32 -compress_type=2 -serialization_prefix ${root_path}/ser/suminc mirror_k=2"  # 阈值1，提前收敛
    echo $cmd
    eval $cmd
done