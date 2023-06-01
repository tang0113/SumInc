
for name in uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
    # update_rate=0.0001
    echo ${name}
    cd /mnt/data/nfs/yusong/dataset/large/${name}/0.0000
    rm -rf /mnt/data/nfs/yusong/dataset/large/${name}/0.0000/${name}_w.base.php
    ls -la -h
    echo -e "-----------------------------\n"
done

# mpirun -n 1 ../build/ingress -application php -vfile /mnt/data/nfs/yusong/dataset/large/road_usa/road_usa.v -efile /mnt/data/nfs/yusong/dataset/large/road_usa/0.0000/road_usa_w.base -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 8 -compress=1 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=23311901 -php_source=23311901 -compress_concurrency=1 -build_index_concurrency=8 -compress_type=2 -serialization_prefix /mnt/data/nfs/yusong/ser/two -serialization_cmp_prefix /mnt/data/nfs/yusong/ser/spnode -message_type=pull


# name=road_usa
# sssp_source=1

# mpirun -n 1 ../build/ingress -application sssp -vfile /mnt/data/nfs/yusong/dataset/large/${name}/${name}.v -efile /mnt/data/nfs/yusong/dataset/large/${name}/0.0000/${name}_w.base -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=0 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=${sssp_source} -php_source=198319 -compress_concurrency=1 -build_index_concurrency=16 -compress_type=2 -serialization_prefix /mnt/data/nfs/yusong/ser/two -serialization_cmp_prefix /mnt/data/nfs/yusong/ser/spnode -message_type=push -verify=0 -out_prefix /mnt/data/nfs/yusong/result/sum_pagerank

# mpirun -n 1 ../build/ingress -application sssp -vfile /mnt/data/nfs/yusong/dataset/large/${name}/${name}.v -efile /mnt/data/nfs/yusong/dataset/large/${name}/0.0000/${name}_w.base -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=1 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=${sssp_source} -php_source=198319 -compress_concurrency=1 -build_index_concurrency=16 -compress_type=2 -serialization_prefix /mnt/data/nfs/yusong/ser/two -serialization_cmp_prefix /mnt/data/nfs/yusong/ser/spnode -message_type=push -verify=1 -out_prefix /mnt/data/nfs/yusong/result/sum_pagerank