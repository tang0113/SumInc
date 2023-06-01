root_path=/mnt2/neu/yusong
# ingress_path=/mnt2/neu/yusong/code/Ingress
ingress_path=/mnt2/neu/yusong/code/SumInc_mirror_expr/SumInc
suminc_path=/mnt2/neu/yusong/code/SumInc_mirror_expr/SumInc
dataset_path=/mnt2/neu/yusong/dataset/expr_data_5
# dataset_path=/mnt2/neu/yusong/dataset/large
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
# verify="-verify=1 "
out_prefix=${root_path}/result
result_name=${app}_4
# ingress_out="-out_prefix ${out_prefix}/Ingress_${result_name}"
# suminc_out="-out_prefix ${out_prefix}/sum_${result_name}_sketch"
compare_result=${root_path}/code/SumInc/expr2/sh
mirror_k=2
min_node_num=2

i=0
max_odegree_id=(15578255 21608483 44499591 16651563 3184732 2164462 6792 20454802 87541572)
min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
source_id=(0 0 0 0 10 0 10 0 10) # high visited
for update_rate in 0.0001 #10 100 1000 10000 100000 100000 1000000
do
for name in webbase-2001 #uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    max_node_num=5000 
    # source=16651563
    i=`expr ${i} + 1`
    build_index_concurrency=$app_concurrency

    #------------------test
    name=uk-2002 # test_mirror_exit europe_osm_small europe_osm_small_1b europe_osm_small_20m
    # max_node_num=5000
    # update_rate=0.0001
    # mirror_k=2
    # min_node_num=2
    source=0 #3 #198319
    # fix= #.random
    # compress_threshold=1 #表示节省的边数, 即：old_inner - index
    # build_index_concurrency=54
    #----------------------

    echo -e "\n\n"  
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    update=
    update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.update"

    efile="-efile ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.base"
    # vfile="-vfile ${dataset_path}/${name}/${name}.v"
    vfile="-vfile ${dataset_path}/${name}/${update_rate}/${name}${fix}.base.v_${max_node_num}"

    # ingress
    # ingress
    cmd="mpirun -n 1 ${ingress_path}/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -portion=1 -sssp_source=${source} -php_source=${source} ${serialization_prefix} -verify=0 ${ingress_out}" # -out_prefix ${out_prefix}/Ingress_${app}
    # echo $cmd
    # eval $cmd
    # echo $ingress_out

    echo -e "\n"

    # suminc
    cmd="mpirun -n 1 ${suminc_path}/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 ${serialization_prefix} ${serialization_cmp_prefix} -compress_threshold=${compress_threshold} -message_type=${type} ${verify} -mirror_k=${mirror_k} ${suminc_out} " # -out_prefix ${out_prefix}/sum_${app}
    echo $cmd
    eval $cmd
    echo $suminc_out
    echo $efile


    # python3 ${compare_result}/compare_result.py ${out_prefix}/sum_pagerank/result_frag_0 ${out_prefix}/Ingress_pagerank/result_frag_0
done
done
# source
# uk-2005: 16651563 /mnt2/neu/yusong/result/sum_sssp_1
# mpirun -n 1 /mnt2/neu/yusong/code/SumInc/build/ingress -application sssp -vfile /mnt2/neu/yusong/dataset/large/google/google.v -efile /mnt2/neu/yusong/result/sum_sssp_sketch/result_frag_0  -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=0 -portion=1 -min_node_num=4 -max_node_num=5000 -sssp_source=16563 -php_source=16563 -compress_concurrency=1 -build_index_concurrency=16 -compress_type=2 -serialization_prefix /mnt2/neu/yusong/ser/suminc -compress_threshold=0 -message_type=push -verify=1 -mirror_k=2  -out_prefix /mnt2/neu/yusong/result/sum_sssp_2


#  mpirun -n 1 /mnt2/neu/yusong/code/Ingress/build/ingress -application sssp -vfile /mnt2/neu/yusong/dataset/large/uk-2002/uk-2002.v -efile /mnt2/neu/yusong/dataset/large/uk-2002/0.0001/uk-2002_w.base -efile_update /mnt2/neu/yusong/dataset/large/uk-2002/0.0001/uk-2002_w.update -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16   -portion=1  -sssp_source=15578255 -php_source=15578255  -verify=0