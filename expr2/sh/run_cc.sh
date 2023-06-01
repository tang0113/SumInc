dataset_path=/mnt/data/nfs/yusong/dataset/large
# dataset_path=/mnt/data/nfs/dataset/
app=cc
ser_type=Ingress
for name in test # uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
do
update_rate=0.0001
# name=test
# update_rate=0.2000

echo -e "\n\n"
time=$(date "+%Y-%m-%d %H:%M:%S")
echo -e "${time}\n"

ext=_ud

update=
update="-efile_update ${dataset_path}/${name}/${update_rate}/${name}${ext}.update"

efile="-efile ${dataset_path}/${name}/${update_rate}/${name}${ext}.base"
vfile="-vfile ${dataset_path}/${name}/${name}.v"
# serialization_cmp_prefix="-serialization_cmp_prefix /mnt/data/nfs/yusong/ser/spnode"
# serialization_prefix="-serialization_prefix /mnt/data/nfs/yusong/ser/${ser_type}"
verify="-verify=1 "
app_concurrency=16
# build_index_concurrency=$app_concurrency
build_index_concurrency=1
type=push
directed=0 # CC algorithm requires undirected graph, run with option: -directed=false


# no-compress
cmd="mpirun -n 1 ../build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=${directed} -cilk=true -termcheck_threshold 0.000001 -app_concurrency ${app_concurrency} -compress=0 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=16651563 -compress_concurrency=1 -build_index_concurrency=${build_index_concurrency} -compress_type=2 ${serialization_prefix}  ${serialization_cmp_prefix} -message_type=${type} -out_prefix /mnt/data/nfs/yusong/result/Ingress_pagerank -verify=1 " # -out_prefix /mnt/data/nfs/yusong/result/Ingress_pagerank --allow-run-as-root
echo $cmd
eval $cmd

# type=pull
cmd="mpirun -n 1 ../build/ingress -application ${app} ${vfile} ${efile} ${update} -directed=${directed} -cilk=true -termcheck_threshold 0.000001 -app_concurrency ${app_concurrency} -compress=1 -portion=1 -min_node_num=5 -max_node_num=1003 -sssp_source=16651563 -compress_concurrency=1 -compress_type=2 ${serialization_cmp_prefix} ${serialization_prefix} -message_type=${type} -out_prefix /mnt/data/nfs/yusong/result/sum_pagerank ${verify} " # -out_prefix /mnt/data/nfs/yusong/result/sum_pagerank
# echo "#备注: separate"
echo "#备注: merge"
echo $cmd
eval $cmd

python3 /mnt/data/nfs/yusong/code/SumInc/expr2/sh/compare_result.py /mnt/data/nfs/yusong/result/sum_pagerank/result_frag_0 /mnt/data/nfs/yusong/result/Ingress_pagerank/result_frag_0
done