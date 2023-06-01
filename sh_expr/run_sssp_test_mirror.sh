root_path=/mnt2/neu/yusong
ingress_path=/mnt2/neu/yusong/code/Ingress
suminc_path=/mnt2/neu/yusong/code/SumInc_mirror_expr/SumInc
# app_root_path=/mnt2/neu/yusong/code/SumInc
dataset_path=/mnt2/neu/yusong/dataset/large
# dataset_path=/mnt2/neu/yusong/dataset/expr_data_5
app=sssp
update_rate=0.0001
type=push
app_concurrency=16
ser_type= #Ingress
weight=_w # _ud_w, 
fix= #.random
termcheck_threshold=0.000001
compress_threshold=1 # 0：表示不过滤, 1：表示内部边大于索引数量
serialization_cmp_prefix="-serialization_cmp_prefix /mnt2/neu/yusong/ser/cluster"
serialization_prefix="-serialization_prefix ${root_path}/ser/suminc"
verify="-verify=0 "
out_prefix=${root_path}/result
result_name=${app}_1
# ingress_out="-out_prefix ${out_prefix}/Ingress_${result_name}"
# suminc_out="-out_prefix ${out_prefix}/sum_${result_name}_sketch"
compare_result=${root_path}/code/SumInc/expr2/sh
mirror_k=200000000 # 模拟不设mirror
min_node_num=4

i=0
max_odegree_id=(15578255 21608483 44499591 16651563 3184732 2164462 6792 20454802 87541572)
min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
source_id=(0 0 0 0 10 0 10 0 10) # high visited


for name in uk-2005 #uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    max_node_num=5000 
    # source=16651563
    i=`expr ${i} + 1`
    build_index_concurrency=$app_concurrency

    #------------------test
    # name=test8 # test_mirror_exit europe_osm_small europe_osm_small_1b europe_osm_small_20m
    # max_node_num=5000
    # update_rate=0.0001
    mirror_k=1000000000
    # min_node_num=2
    # source=0 #3 #198319
    # fix= #.random
    compress_threshold=-10000000 #表示节省的边数, 即：old_inner - index
    build_index_concurrency=54
    #----------------------

    echo -e "\n\n"  
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    update=
    # update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.update"

    efile="-efile ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.base"
    vfile="-vfile ${dataset_path}/${name}/${name}.v"
    # vfile="-vfile ${dataset_path}/${name}/${update_rate}/${name}${fix}.base.v_${max_node_num}"


    echo -e "\n"

    # suminc
    cmd="mpirun -n 1 ${suminc_path}/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 ${serialization_prefix} ${serialization_cmp_prefix} -compress_threshold=${compress_threshold} -message_type=${type} ${verify} -mirror_k=${mirror_k} ${suminc_out} " # -out_prefix ${out_prefix}/sum_${app}
    echo $cmd
    eval $cmd
    echo $suminc_out
    echo $efile


    # python3 ${compare_result}/compare_result.py ${out_prefix}/sum_pagerank/result_frag_0 ${out_prefix}/Ingress_pagerank/result_frag_0
done