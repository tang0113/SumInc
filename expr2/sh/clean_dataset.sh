echo "clean dataset..."
for name in hollywood-2009 uk-2002 uk-2005 europe_osm road_usa friendster google soc-livejournal delaunay_n24 coAuthorsDBLP hollywood-2009 soc-orkut com-friendster soc-twitter-2010 roadNet-CA
do
    for rate in 0.010 0.001
    do
        rm -rf /mnt/data/nfs/yusong/dataset/large/${name}/${name}_${rate}.v
        rm -rf /mnt/data/nfs/yusong/dataset/large/${name}/${name}_w_${rate}.v
        rm -rf /mnt/data/nfs/yusong/dataset/large/${name}/${rate}
    done
done