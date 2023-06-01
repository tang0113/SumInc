# 根据聚类文件,生成需要的点文件

# eq: g++ gen_v_file.cc -o gen_v_file && ./gen_v_file /mnt2/neu/yusong/dataset/large/test8/0.0001/test8.base.c_5000 /mnt2/neu/yusong/dataset/large/test8/0.0001/test8.v 26

root_path=/mnt2/neu/yusong
app_root_path=/mnt2/neu/yusong
max_node_num=5000
update_rate=0.0001
fix= #.random

g++ ${app_root_path}/code/SumInc/expr2/utils/gen_v_file.cc -o ${app_root_path}/code/SumInc/expr2/utils/gen_v_file

# 18484117	39454746	50912018	23947347	22743892	41290682	30809122	11546657	115657290
node_nums=(23947347) #26 18484117	39454746	50912018	23947347	22743892	41290682	30809122	11546657	115657290)
i=0
for name in road_usa #test8 uk-2002	uk-2005	europe_osm	road_usa	arabic-2005	it-2004	gsh-2015-tpd	sk-2005	webbase-2001
do
  node_num=${node_nums[${i}]}
  i=`expr ${i} + 1`
  efile="${root_path}/dataset/large/${name}/${update_rate}/${name}${fix}.base"
  cmd="${app_root_path}/code/SumInc/expr2/utils/gen_v_file ${efile}.c_${max_node_num} ${efile}.v_${max_node_num} ${node_num}"
  echo $cmd
  eval $cmd
done
