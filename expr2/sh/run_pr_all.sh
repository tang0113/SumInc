for thread_num in 1 8 16 #32
do
for message_type in push pull
do
    for name in uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
    do
        update_rate=0.0000

        echo -e "\n\n"
        time=$(date "+%Y-%m-%d %H:%M:%S")
        echo -e "${time}\n"

        update=
        # update="-efile_update /mnt/data/nfs/yusong/dataset/large/${name}/${update_rate}/${name}.update"

        app_concurrency=$thread_num
        build_index_concurrency=$app_concurrency
        type=$message_type

        # inc
        cmd="mpirun -n 1 ../build/ingress -application pagerank -vfile /mnt/data/nfs/yusong/dataset/large/${name}/${name}.v -efile /mnt/data/nfs/yusong/dataset/large/${name}/${update_rate}/${name}.base  ${update} -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=16651563 -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 -serialization_prefix /mnt/data/nfs/yusong/ser/Ingress -serialization_cmp_prefix /mnt/data/nfs/yusong/ser/spnode -message_type=${type}" # -out_prefix /mnt/data/nfs/yusong/result/sum_pagerank
        # echo "#备注: separate"
        echo "#备注: merge"
        echo $cmd
        eval $cmd
    done

    for name in uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
    do
        update_rate=0.0000

        echo -e "\n\n"
        time=$(date "+%Y-%m-%d %H:%M:%S")
        echo -e "${time}\n"

        update=
        # update="-efile_update /mnt/data/nfs/yusong/dataset/large/${name}/${update_rate}/${name}.update"

        app_concurrency=$thread_num
        build_index_concurrency=$app_concurrency
        type=$message_type

        # no-compress
        cmd="mpirun -n 1 ../build/ingress -application pagerank -vfile /mnt/data/nfs/yusong/dataset/large/${name}/${name}.v -efile /mnt/data/nfs/yusong/dataset/large/${name}/${update_rate}/${name}.base ${update} -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency ${app_concurrency} -compress=0 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=16651563 -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 -serialization_prefix /mnt/data/nfs/yusong/ser/Ingress -serialization_cmp_prefix /mnt/data/nfs/yusong/ser/spnode -message_type=${type} " # -out_prefix /mnt/data/nfs/yusong/result/Ingress_pagerank  --allow-run-as-root
        echo $cmd
        eval $cmd

    done

done
done