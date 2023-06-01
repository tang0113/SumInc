root_path=/mnt2/neu/yusong
app_root_path=/mnt2/neu/yusong
app=sssp
update_rate=0.0001
type=push
app_concurrency=16
ser_type= #Ingress
weight=_w_ud # _w_ud, 
fix=.random
termcheck_threshold=0.000001
compress_threshold=1 # 0：表示不过滤, 1：表示内部边大于索引数量
serialization_prefix="-serialization_prefix ${root_path}/ser/grape"
verify="-verify=1 "
out_prefix=${root_path}/result
result_name=${app}_1
grape_out="-out_prefix ${out_prefix}/grape_${result_name}_sketch"
compare_result=${root_path}/code/SumInc/expr2/sh

i=0
max_odegree_id=(198319 23311901 44499591 16651563 3907147 2164462 6792 0 89445946)
min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
source_id=(198319 23311901 44499591 16651563 3907147 2164462 6792 0 89445946)
for name in road_usa #uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    source=${source_id[${i}]}
    # source=16651563
    i=`expr ${i} + 1`

    #------------------test
    name=google # test_mirror_exit
    update_rate=0.0001
    source=3 #198319
    fix= #.random
    #----------------------

    echo -e "\n\n"  
    time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${time}\n"

    updated="${root_path}/dataset/large/${name}/${update_rate}/${name}${weight}${fix}.updated"

    vfile="-vfile ${root_path}/dataset/large/${name}/${name}.v"

    # suminc
    cmd="mpirun -n 1 ${app_root_path}/code/libgrape-lite/build/run_app --application ${app} ${vfile} --efile ${updated} --sssp_source ${source} --directed=true --termcheck_threshold=0.00000001 --segmented_partition=false -rebalance=false --app_concurrency=16 ${grape_out} ${serialization_prefix}" # -out_prefix ${out_prefix}/sum_${app}
    echo $cmd
    eval $cmd
    echo $grape_out

    # python3 ${compare_result}/compare_result.py ${out_prefix}/sum_pagerank/result_frag_0 ${out_prefix}/Ingress_pagerank/result_frag_0
done
