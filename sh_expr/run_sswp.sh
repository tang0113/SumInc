root_path=/mnt2/neu/yusong
ingress_path=/mnt2/neu/yusong/code/Ingress
suminc_path=/mnt2/neu/yusong/code/SumInc
dataset_path=/mnt2/neu/yusong/dataset/large
app=sswp
update_rate=0.0001
type=push
app_concurrency=1
ser_type= #Ingress
weight=_w # _ud_w, 
fix= #.random
termcheck_threshold=0.000001
compress_threshold=1 # 0：表示不过滤, 1：表示内部边大于索引数量
serialization_cmp_prefix="-serialization_cmp_prefix /mnt2/neu/yusong/ser/cluster"
serialization_prefix="-serialization_prefix ${root_path}/ser/suminc"
# verify="-verify=1 "
out_prefix=${root_path}/result
result_name=${app}_2
mirror_k=2
min_node_num=2

i=0
max_odegree_id=(15578255 21608483 44499591 16651563 3184732 2164462 6792 20454802 87541572)
min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
source_id=(0 0 0 0 10 0 10 0 10) # high visited
for name in test8 #uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    max_node_num=5000 
    # source=16651563
    i=`expr ${i} + 1`
    build_index_concurrency=$app_concurrency

    ------------------test
    name=google # test_mirror_exit europe_osm_small europe_osm_small_1b europe_osm_small_20m
    max_node_num=5000
    update_rate=0.0001
    mirror_k=200000
    min_node_num=2
    source=10000 #3 #198319
    fix= #.random
    compress_threshold=1222 #表示节省的边数, 即：old_inner - index, 如果测试不压缩，可以将其设置为无穷大
    # build_index_concurrency=54
    #----------------------

    echo -e "\n\n"  
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    update=
    update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.update"

    efile="-efile ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.base"
    vfile="-vfile ${dataset_path}/${name}/${name}.v"
    # vfile="-vfile ${dataset_path}/${name}/${update_rate}/${name}${fix}.base.v_${max_node_num}"
    ingress_out="-out_prefix ${out_prefix}/Ingress_${result_name}_min${min_node_num}"
    suminc_out="-out_prefix ${out_prefix}/sum_${result_name}_min${min_node_num}"
    compare_result=${root_path}/code/SumInc/expr2/sh

    # ingress
    cmd="mpirun -n 1 ${ingress_path}/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -portion=1 -sssp_source=${source} -php_source=${source} ${serialization_prefix} -verify=0 ${ingress_out}" # -out_prefix ${out_prefix}/Ingress_${app}
    echo $cmd
    eval $cmd
    echo $ingress_out

    echo -e "\n"

    # suminc
    cmd="mpirun -n 1 ${suminc_path}/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 ${serialization_prefix} ${serialization_cmp_prefix} -compress_threshold=${compress_threshold} -message_type=${type} ${verify} -mirror_k=${mirror_k} ${suminc_out} " # -out_prefix ${out_prefix}/sum_${app}
    echo $cmd
    eval $cmd
    echo $suminc_out
    echo $efile

    check="python3 ${compare_result}/compare_result.py ${out_prefix}/sum_${result_name}_min${min_node_num}/result_frag_0 ${out_prefix}/Ingress_${result_name}_min${min_node_num}/result_frag_0"
    echo $check
    eval $check
done