#
# 为graphbolt生成数据集
#   只需要将base生成好即可
# 注意： 只处理无权图数据集
#
root_path=/mnt2/neu/yusong
cd ${root_path}/code/GraphBolt/
cd tools/converters
# make -j
# file=/mnt2/neu/yusong/dataset/large/road_usa/0.0001/road_usa_w_ud
# ./SNAPtoAdjConverter /mnt2/neu/yusong/dataset/large/road_usa/0.0001/road_usa_w_ud.base /mnt2/neu/yusong/dataset/large/road_usa/0.0001/road_usa_w_ud.adj

update_rate=0.0001
weight=
fix=
for name in webbase-2001 # 数据集名称
do
  efile="${root_path}/dataset/large/${name}/${update_rate}/${name}${weight}${fix}.base"
  ./SNAPtoAdjConverter ${efile} ${efile}.adj
done