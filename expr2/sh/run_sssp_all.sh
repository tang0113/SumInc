dataset_path=/mnt/data/nfs/yusong/dataset/large
app=sssp
ser_type=Ingress
# app=pagerank
# ser_type=one
for app_concurrency in 16 # 1 8 16 #32
do 
for message_type in push #push pull
do
    # max_odegree_id=(198319 23311901 44499591 16651563 3907147 2164462 6792 0 89445946) # real
    max_odegree_id=(198319 23311901 10000 16651563 0 0 6792 0 1000) # modified
    min_odegree_Id=(448 2898 31885 2876 22285696 3604 17371 0 2712)
    list_names=(uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001)
    # list_names=(uk-2002 uk-2005)
    i=0
    for name in ${list_names[@]}
    do
        update_rate=0.0000
        source=${max_odegree_id[${i}]}
        i=`expr ${i} + 1`
        build_index_concurrency=$app_concurrency

        echo -e "\n\n"
        time=$(date "+%Y-%m-%d %H:%M:%S")
        echo -e "${time}\n"

        update=
        # update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}_w.update"

        efile="-efile ${dataset_path}/${name}/${update_rate}/${name}_w.base"
        vfile="-vfile ${dataset_path}/${name}/${name}.v"
        serialization_prefix=/mnt/data/nfs/yusong
        out_prefix=/mnt/data/nfs/yusong/result

        cmd="mpirun -n 1 ../build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency ${app_concurrency} -compress=0 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 -serialization_prefix ${serialization_prefix}/ser/${ser_type}  -serialization_cmp_prefix ${serialization_prefix}/ser/spnode -message_type=${message_type} -verify=0 " #-out_prefix ${out_prefix}/Ingress_pagerank
        echo $cmd
        eval $cmd
    done

    i=0
    for name in ${list_names[@]}
    do
        update_rate=0.0000
        source=${max_odegree_id[${i}]}
        i=`expr ${i} + 1`
        build_index_concurrency=$app_concurrency

        echo -e "\n\n"
        time=$(date "+%Y-%m-%d %H:%M:%S")
        echo -e "${time}\n"

        update=
        # update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}_w.update"

        efile="-efile ${dataset_path}/${name}/${update_rate}/${name}_w.base"
        vfile="-vfile ${dataset_path}/${name}/${name}.v"
        serialization_prefix=/mnt/data/nfs/yusong
        out_prefix=/mnt/data/nfs/yusong/result

        cmd="mpirun -n 1 ../build/ingress -application ${app} ${vfile} ${efile} ${update} ${update} -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=${source} -php_source=${source} -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 -serialization_prefix ${serialization_prefix}/ser/${ser_type}  -serialization_cmp_prefix ${serialization_prefix}/ser/spnode -message_type=${message_type} -verify=0 " # -out_prefix ${out_prefix}/sum_pagerank
        echo $cmd
        eval $cmd
    done

done
done
