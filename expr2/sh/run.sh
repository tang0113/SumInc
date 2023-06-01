# cd ../Release
cd ../../build
# for name in road_usa
for name in soc-twitter-2010
do
for max_node_num in 1003 #1004 #1001时选按照入度由小到大，出度由小到大的顺序选择。1002 采样, 1003 obj=1.2, 1005局部最优
do
# name=road_usa
# name=europe_osm
# name=uk-2005
# name=soc-twitter-2010
# name=com-friendster
# name=web-uk-2002-all
# name=coAuthorsDBLP, 0.151354(scan++)
# name=belgium_osm, 0.0138473(scan++)
# name=ak2010, scan++不如mode2
# percentage=0.010
percentage=0.0001
# path=/mnt/data/nfs/yusong/dataset/${name}
# path=/mnt/data/nfs/dataset/${name}
path=/mnt/data/nfs/yusong/dataset/large/${name}

termcheck_threshold=0.000001
# termcheck_threshold=0.01
work_num=1
app_concurrency=1 # 代码中默认值为：-1
compress=0 # 是否压缩
compress_concurrency=1    # 并行压缩线程数量, -1表示不用多线程
build_index_concurrency=52 # 计算索引时的线程数量
cilk=true
compress_type=2 # 0:mode2, 1:metis, 2:scan++
portion=1
min_node_num=5
# max_node_num=8
directed=1

if [ ${directed} -eq 0 ];then
    app=pagerank
    serialization_prefix=/mnt/data/nfs/yusong/ser/one
    efile_update=${path}/${percentage}/${name}_ud.update
    efile_updated=${path}/${percentage}/${name}_ud.updated
    efile=${path}/${percentage}/${name}_ud.base
    vfile=${path}/${name}.v
else
    app=pagerank
    serialization_prefix=/mnt/data/nfs/yusong/ser/one
    efile_update=${path}/${percentage}/${name}.update
    efile_updated=${path}/${percentage}/${name}.updated
    efile=${path}/${percentage}/${name}.base
    # efile=${path}/${name}.e # 源文件
    vfile=${path}/${name}.v
fi
# app=sssp
# serialization_prefix=/mnt/data/nfs/yusong/ser/two
# efile_update=${path}/${percentage}/${name}_w.update
# efile_updated=${path}/${percentage}/${name}_w.updated
# efile=${path}/${percentage}/${name}_w.base
# vfile=${path}/${name}_${percentage}.v

echo -e "\n\n"

serialization_cmp_prefix=/mnt/data/nfs/yusong/ser/spnode
out_result=/mnt/data/nfs/yusong/result
echo $efile
echo $efile_update
echo $vfile

# 序列化
# 第一次运行先需要用-serialize -serialization_prefix xxxx序列化

# 压缩
# # run  -segmented_partition=true 
compress=1
cmd="mpirun -n $work_num ./ingress -application ${app} -vfile ${vfile} -efile ${efile} -efile_update ${efile_update} -directed=1 -cilk=${cilk} -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency}  -compress=${compress} -portion=${portion} -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=16651563 -compress_concurrency=${compress_concurrency} -build_index_concurrency=${build_index_concurrency} -compress_type=${compress_type} -serialization_cmp_prefix=${serialization_cmp_prefix} -serialization_prefix ${serialization_prefix} -out_prefix ${out_result}/sum_${app}" # -efile_update ${efile_update}
echo $cmd
eval $cmd

# # 不压缩
echo -e "\n"
compress=0
cmd="mpirun -n $work_num ./ingress -application ${app} -vfile ${vfile} -efile ${efile} -efile_update ${efile_update} -directed=1 -cilk=${cilk} -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency}  -compress=0 -portion=${portion} -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=16651563 -compress_concurrency=${compress_concurrency} -build_index_concurrency=${build_index_concurrency} -compress_type=${compress_type} -serialization_cmp_prefix=${serialization_cmp_prefix} -serialization_prefix ${serialization_prefix} -out_prefix ${out_result}/Ingress_${app}" # -efile_update ${efile_update}
echo $cmd
eval $cmd

# ingress
# mpirun -n $work_num /home/neu/code/z_Ingress/build/ingress -application ${app} -efile ${efile} -efile_update ${efile_update} -vfile ${vfile}  -out_prefix ${out_result}/Ingress_${app} -app_concurrency ${app_concurrency} -termcheck_threshold ${termcheck_threshold} -sssp_source=0 #-cilk=1
# 

# non-inc
# cmd="mpirun -n $work_num ./ingress -application ${app} -vfile ${vfile} -efile ${efile} -cilk=${cilk} -termcheck_threshold ${termcheck_threshold} -app_concurrency ${app_concurrency}  -compress=${compress} -portion=${portion} -min_node_num=${min_node_num} -max_node_num=${max_node_num} -sssp_source=16651563 -compress_concurrency=${compress_concurrency} -build_index_concurrency=${build_index_concurrency} -compress_type=${compress_type} -serialization_cmp_prefix=${serialization_cmp_prefix} -serialization_prefix ${serialization_prefix} -out_prefix ${out_result}/sum_${app}" # -efile_update ${efile_update}
# echo $cmd
# eval $cmd

echo "start check..."
python3 /mnt/data/nfs/yusong/code/SumInc/expr2/sh/compare_result.py ${out_result}/sum_${app}/result_frag_0 ${out_result}/Ingress_${app}/result_frag_0

# analyze data
# echo "analyze data..."
# python3 extract_result_2.py log_test
done
done