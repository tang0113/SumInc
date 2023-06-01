# 生成增量计算所需要的数据集
#    有向图/无向图/带权图/无权图
# 
root_path=/mnt2/neu/yusong

# directed=_ud #无向图
directed=    # 有向图
# head=-header # 有head
head=        # 无head
for name in test
do # {
    # efile=/mnt/data/nfs/yusong/dataset/large/${name}/${name}.base
    for percentage in 0.1 #0.0100 #0.001 0.0001
    do
        for weight in 0 1
        do 
            python3 ${root_path}/code/SumInc/expr2/sh/gen_inc2.py ${root_path}/dataset/large/${name}/${name}${directed}.e ${percentage} -w=${weight} ${head} 
            # python3 ${root_path}/code/SumInc/expr2/sh/gen_inc2_old.py ${root_path}/dataset/large/${name}/${name}${directed}.e ${percentage} -w=${weight} ${head}
        
        done
    done
done #} &
# wait
# dataset: uk-2002 uk-2005 arabic-2005 europe_osm road_usa friendster google soc-livejournal delaunay_n24 coAuthorsDBLP hollywood-2009 soc-orkut com-friendster soc-twitter-2010