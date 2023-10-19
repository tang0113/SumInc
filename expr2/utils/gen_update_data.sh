root_path=/home/tjn/test/dataset/large/test
app_root_path=/home/tjn/tjnCode/SumInc-count_active_edge2/expr2/utils

inc_rate=0.0001
fix=.e
rand=5
for inc_rate in 0.0001
do
  for name in uk-2005 #uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001 # uk-2002 
  do
      cmd="g++ ${app_root_path}/deal_dataset.cc -lgflags -lglog -o ${app_root_path}/deal_dataset && ${app_root_path}/deal_dataset -weight=0 -base_edge=${root_path}/${name}/${name}${fix} -out_edge=${root_path}/${name}/${inc_rate}/${name}.update -out_dir=${root_path}/${name} -inc_rate=${inc_rate} -hava_inadj=1 -head=0" # 必须手动配置head
      echo $cmd
      eval $cmd

      # cp ${root_path}/dataset/large/${name}/${name}.e ${root_path}/dataset/expr_data/${name}/${name}.e
      # cp ${root_path}/dataset/large/${name}/${name}.e.b ${root_path}/dataset/expr_data/${name}/${name}.e.b
      # ls ${root_path}/dataset/expr_data/${name}
  done
done

# 制作完数据集，开始压缩
sh /mnt2/neu/yusong/code/SumInc/sh_expr/louvain_all.sh


# for inc_rate in 0.001 10000
# do
#   for name in uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
#   do
#     ipath=${root_path}/dataset/expr_data/${name}/${inc_rate}/${name}${fix}.base
#     opath=${root_path}/dataset/louvain_bin/expr_data_${name}_${inc_rate}${fix}
#     louvain_path=${root_path}/code/louvain/gen-louvain
#     # 格式转换
#     ${louvain_path}/convert -i ${ipath} -o ${opath}.bin
#   done
# done


# fix=_ud
# for inc_rate in 0.0010 10000
# do
#   for name in uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
#   do
#       cmd="g++ ${app_root_path}/deal_dataset_ud.cc -lgflags -lglog -o ${app_root_path}/deal_dataset_ud && ${app_root_path}/deal_dataset_ud -weight=0 -base_edge=${root_path}/dataset/expr_data/${name}/${name} -out_edge=${root_path}/dataset/expr_data/${name}/${inc_rate}/${name}${fix} -out_dir=/mnt2/neu/yusong/dataset/expr_data/${name}/ -inc_rate=${inc_rate} -hava_inadj=1 -head=0" # 必须手动配置head
#       echo $cmd
#       eval $cmd

#       # cp ${root_path}/dataset/large/${name}/${name}.e ${root_path}/dataset/expr_data/${name}/${name}.e
#       # cp ${root_path}/dataset/large/${name}/${name}.e.b ${root_path}/dataset/expr_data/${name}/${name}.e.b
#       # ls ${root_path}/dataset/expr_data/${name}
#   done
# done

# for inc_rate in 0.0010 10000
# do
#   for name in uk-2002 uk-2005 europe_osm road_usa arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001
#   do
#     ipath=${root_path}/dataset/expr_data/${name}/${inc_rate}/${name}${fix}.base
#     opath=${root_path}/dataset/louvain_bin/expr_data_${name}_${inc_rate}${fix}
#     louvain_path=${root_path}/code/louvain/gen-louvain
#     # 格式转换
#     ${louvain_path}/convert -i ${ipath} -o ${opath}.bin
#   done
# done

## head: road_usa: 163
## head: europe_osm: 3
## uk-2002 uk-2005 webbase-2001 arabic-2005 it-2004
# rm /mnt2/neu/yusong/dataset/louvain_bin/expr_data_webbase-2001_0.0001.bin
# []
# mpirun -n 1 /mnt2/neu/yusong/code/SumInc/build/ingress -application sssp -vfile /mnt2/neu/yusong/dataset/expr_data/gsh-2015-tpd/gsh-2015-tpd.v -efile /mnt2/neu/yusong/dataset/expr_data/road_usa/10000/road_usa_ud_w.base -efile_update /mnt2/neu/yusong/dataset/expr_data/road_usa/10000/road_usa_ud_w.update -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=0 -portion=1 -min_node_num=5 -max_node_num=5000 -sssp_source=0 -php_source=0 -compress_concurrency=1 -build_index_concurrency=16 -compress_type=2 -serialization_prefix /mnt2/neu/yusong/ser/suminc -compress_threshold=1 -message_type=push -verify=0
