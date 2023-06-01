python_path=/mnt2/neu/yusong/code/SumInc/expr2/sh
dataset_path=/mnt2/neu/yusong/dataset/large
# .e, 生成数据集
echo "deal .e/.base dataset..."
for name in webbase-2001 # uk-2005 arabic-2005 it-2004 gsh-2015-tpd sk-2005 webbase-2001 #europe_osm
do
    # tar -C ${dataset_path}/${name} -xzvf ${dataset_path}/${name}.tar.gz
    for rate in 1000 10000
    do
    {
        for weight in 1 0 #1 # 权
        do
        {
            python3 ${python_path}/gen_inc2_batch_random.py ${dataset_path}/${name}/${name}.e ${rate} -w=${weight} 
        } #&
        done
    } #&
    done
done
wait

# .mtx, 文件中带有注释(%/#)， 非注释第一行为：顶点数，顶点数，边数
# echo "deal .mtx dataset..."
# for name in road_usa europe_osm # road_usa # uk-2002
# do
#     for rate in 1000 10000 #0.01 0.001 0.0001
#     do
#     {
#         for weight in 1 0 # 1 # weight
#         do
#         {
#             cmd="python3 ${python_path}/gen_inc2_batch_random.py ${dataset_path}/${name}/${name}.mtx ${rate} -w=${weight} -header"
#             echo $cmd
#             eval $cmd
#         } &
#         done
#     } #&
#     done
# done
# wait  ##等待所有子后台进程结束

# .mtx文件转.e文件
# echo "deal .mtx dataset..."
# for name in indochina-2004 europe_osm delaunay_n24 cit-Patents soc-LiveJournal1 arabic-2005 soc-orkut uk-2005
# do
#     python3 ${python_path}/mtx2e.py ${dataset_path}/${name}/${name}.mtx -header
# done

# dataset: arabic-2005 uk-2002 uk-2005 europe_osm road_usa friendster google soc-LiveJournal1 delaunay_n24 coAuthorsDBLP hollywood-2009 soc-orkut com-friendster soc-twitter-2010 roadNet-CA webbase-2001 gsh-2015-tpd it-2004
# Note： mtx数据集，前几行是以%或者#开头的注释，非注释行第一行为三个数，分别表示点数，点数，边数.
# dataset name:     点数，    点数，    边数,       大小，可视化网址
# indochina-2004:   7414866  7414866  194109311
# hollywood-2009:   1139905  1139905  57515616
# europe_osm:       50912018 50912018 54054660
# delaunay_n24:     16777216 16777216 50331601   801M https://networkrepository.com/delaunay-n10.php
# coAuthorsDBLP:    299067   299067   977676
# cit-Patents:      3774768  3774768  16518948
# belgium_osm:      1441295  1441295  1549970
# asia_osm:         11950757 11950757 12711603   196M
# ak2010:           45292    45292    108549
# soc-LiveJournal1: 4847571  4847571  68993773   65M
# soc-orkut:        2997166  2997166  106349209
# arabic-2005:      22744080 22744080 639999458  11G
# soc-sinaweibo:    58655849 58655849 261321071  3.6G
# soc-twitter-2010: 21297772 21297772 265025809  4.9G
# soc-twitter:  23G
# uk-2002:          8520486  18520486 298113762
# uk-2005:          39459925 39459925 936364282
# friendster: 795M
# com-firendster:   65608366 65608366 1806067135 
# webbase-2001:                                  18G      
# gsh-2015-tpd:                                  9.0G
# it-2004:          41291594 41291594 1150725436 19G