root_path=/mnt2/neu/yusong
app_root_path=/mnt2/neu/yusong
dataset_path=${root_path}/dataset/expr_data
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
verify="-verify=1 "
out_prefix=${root_path}/result
result_name=${app}_1
# ingress_out="-out_prefix ${out_prefix}/Ingress_${result_name}"
# suminc_out="-out_prefix ${out_prefix}/sum_${result_name}_sketch"
# oingress_out="-out_prefix ${out_prefix}/oIngress_${result_name}"
# compare_result=${root_path}/code/SumInc/expr2/sh
mirror_k=4
min_node_num=5

i=0
max_odegree_id=(15578255 21608483 44499591 16651563 3184732 2164462 6792 20454802 87541572)
min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
source_id=(15578255 21608483 44499591 16651563 10000 10000 100000 20454802 10000) # high visited
for name in test8 #uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    max_node_num=5000 
    # source=16651563
    i=`expr ${i} + 1`
    build_index_concurrency=$app_concurrency

    #------------------test
    name=uk-2002 # test_mirror_exit
    max_node_num=5000
    update_rate=0.0001
    mirror_k=2
    min_node_num=2
    source=0 #198319
    fix= #.random
    compress_threshold=0 #表示节省的边数, 即：old_inner - index
    # build_index_concurrency=54
    #----------------------

    echo -e "\n\n"  
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    update=
    update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.update"

    efile="-efile ${dataset_path}/${name}/${update_rate}/${name}${weight}${fix}.base"
    vfile="-vfile ${dataset_path}/${name}/${name}.v"

    # new_ingress
    cmd="mpirun -n 1 ${app_root_path}/code/SumInc/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -compress=0 -portion=1 -min_node_num=5 -max_node_num=${max_node_num} -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 ${serialization_prefix} -compress_threshold=${compress_threshold} -message_type=${type} -verify=1 ${ingress_out}" # -out_prefix ${out_prefix}/Ingress_${app}
    # echo $cmd
    # eval $cmd
    # echo $ingress_out

    echo -e "\n"

    # suminc
    cmd="mpirun -n 1 ${app_root_path}/code/SumInc/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 ${serialization_prefix} ${serialization_cmp_prefix} -compress_threshold=${compress_threshold} -message_type=${type} ${verify} -mirror_k=${mirror_k} ${suminc_out} " # -out_prefix ${out_prefix}/sum_${app}
    # echo $cmd
    # eval $cmd
    # echo $suminc_out

    # origin_ingress
    serialization_prefix="-serialization_prefix ${root_path}/ser/oIngress"
    cmd="mpirun -n 1 ${app_root_path}/code/Ingress/build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency} -portion=1 -sssp_source=${source} -php_source=${source} ${serialization_prefix} -verify=1 ${oingress_out}" # -out_prefix ${out_prefix}/Ingress_${app}
    echo $cmd
    eval $cmd
    echo $oingress_out

    echo -e "\n"

    # python3 ${compare_result}/compare_result.py ${out_prefix}/sum_pagerank/result_frag_0 ${out_prefix}/Ingress_pagerank/result_frag_0
done

# source
# uk-2005: 16651563 /mnt2/neu/yusong/result/sum_sssp_1
# mpirun -n 1 /mnt2/neu/yusong/code/SumInc/build/ingress -application sssp -vfile /mnt2/neu/yusong/dataset/large/google/google.v -efile /mnt2/neu/yusong/result/sum_sssp_sketch/result_frag_0  -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=0 -portion=1 -min_node_num=4 -max_node_num=5000 -sssp_source=16563 -php_source=16563 -compress_concurrency=1 -build_index_concurrency=16 -compress_type=2 -serialization_prefix /mnt2/neu/yusong/ser/suminc -compress_threshold=0 -message_type=push -verify=1 -mirror_k=2  -out_prefix /mnt2/neu/yusong/result/sum_sssp_2