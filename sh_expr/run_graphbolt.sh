#
# 运行 GraphBolt的pgarank
#

cd /mnt2/neu/yusong/code_test/GraphBolt/
root_path=/mnt2/neu/yusong
app_root_path=/mnt2/neu/yusong
app=PageRank # PageRank SSSP
update_rate=0.0001
type=push
app_concurrency=16
ser_type= #Ingress
weight= # _w_ud, 
fix=
termcheck_threshold=0.000001
compress_threshold=1 # 0：表示不过滤, 1：表示内部边大于索引数量
serialization_cmp_prefix="-serialization_cmp_prefix /mnt2/neu/yusong/ser/cluster"
serialization_prefix="-serialization_prefix ${root_path}/ser/suminc"
verify="-verify=1 "
out_prefix=${root_path}/result
compare_result=${root_path}/code/SumInc/expr2/sh
mirror_k=4
min_node_num=5

i=0
max_odegree_id=(198319 23311901 44499591 16651563 3907147 2164462 6792 0 89445946)
min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
source_id=(0)
for name in uk-2005 #uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    # source=16651563
    i=`expr ${i} + 1`

    echo -e "\n\n"
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    update=
    update="${root_path}/dataset/large/${name}/${update_rate}/${name}${weight}${fix}.update"

    efile="${root_path}/dataset/large/${name}/${update_rate}/${name}${weight}${fix}.base"
    #efile="${root_path}/dataset/large/${name}/${update_rate}/${name}${fix}.base"
    result_name=${app}_${name}

    # 输出
    ingress_out="-out_prefix ${out_prefix}/Ingress_${result_name}_min${min_node_num}"
    suminc_out="-out_prefix ${out_prefix}/sum_${result_name}_min${min_node_num}"
    check="python3 ${compare_result}/compare_result.py ${out_prefix}/sum_${result_name}_min${min_node_num}/result_frag_0 ${out_prefix}/Ingress_${result_name}_min${min_node_num}/result_frag_0"

    # wc -l ${efile} # 获取边文件的行数， 填充到下面的 -nEdges参数
    result=$(wc -l ${efile})
    edge_num=$(echo $result | grep -P '\d+ ' -o)
    echo $result
    echo "edge_num=${edge_num}"
    cmd="LD_PRELOAD=/mnt2/neu/yusong/code/GraphBolt/lib/mimalloc/out/release/libmimalloc.so /mnt2/neu/yusong/code/GraphBolt/apps/${app} -numberOfUpdateBatches 1 -nEdges ${edge_num} -maxIters 100 -streamPath ${update} -source=0 -nWorkers 16 ${efile}.adj"
    echo $cmd
    eval $cmd

done