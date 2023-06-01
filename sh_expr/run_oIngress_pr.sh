root_path=/mnt2/neu/yusong
app_root_path=/mnt2/neu/yusong
app=pagerank
update_rate=0.0001
type=push
app_concurrency=16
ser_type= #Ingress
weight= # _w_ud, 
fix=.random
termcheck_threshold=0.000001
compress_threshold=1 # 0：表示不过滤, 1：表示内部边大于索引数量
# serialization_cmp_prefix="-serialization_cmp_prefix /mnt2/neu/yusong/ser/spnode"
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
for name in sk-2005 #uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    max_node_num=5000 
    # source=16651563
    i=`expr ${i} + 1`
    build_index_concurrency=$app_concurrency

    #------------------test
    name=uk-2005 # test_mirror_exit google
    max_node_num=5000
    update_rate=0.0001
    mirror_k=2
    min_node_num=2  # P.size() >= MIN_NODE_NUM, 这个越大,包含的点越少,校正阶段越久
    source=0 #198319, -1表示没有源点，即
    fix= #.random
    compress_threshold=0 #表示节省的边数, 即：old_inner - index
    # build_index_concurrency=54
    #----------------------

    echo -e "\n\n"
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    update=
    update="-efile_update ${root_path}/dataset/large//${name}/${update_rate}/${name}${weight}.update"

    efile="-efile ${root_path}/dataset/large/${name}/${update_rate}/${name}${weight}.base"
    vfile="-vfile ${root_path}/dataset/large/${name}/${name}.v"
    result_name=${app}_${name}
    # suminc_out="-out_prefix ${out_prefix}/sum_${result_name}_min${min_node_num}"

    # ingress
    # ingress_out="-out_prefix ${out_prefix}/oIngress_${result_name}_min${min_node_num}"
    serialization_prefix="-serialization_prefix ${root_path}/ser/oIngress"
    cmd="mpirun -n 1 ${app_root_path}/code/Ingress/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -portion=1 -sssp_source=${source} -php_source=${source} ${serialization_prefix}  ${ingress_out}" # -out_prefix ${out_prefix}/Ingress_pagerank
    echo $cmd
    eval $cmd

    # python3 ${compare_result}/compare_result.py ${out_prefix}/sum_${result_name}_min${min_node_num}/result_frag_0 ${out_prefix}/Ingress_${result_name}_min${min_node_num}/result_frag_0
done