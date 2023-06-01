ext=('base' 'update' 'updated')
root=/mnt2/neu/yusong

dataset=('web-google')
for name in "${dataset[@]}"
do 
{
    echo "Generating $name"
    for ext in "${ext[@]}"
    do
        # for p in 0.010 0.001
        for p in 0.0001 0.0100 0.0010
        do
            # python3 d2ud.py ${path}/${name}/${p}/${name}.${ext}
            python3 ${root}/code/SumInc/expr2/sh/d2ud.py ${root}/dataset/large/${name}/${p}/${name}_w.${ext}
        done
    done
} &
done
wait
# dataset=('uk-2002')
# path=/mnt/data/nfs/yusong/dataset/large
# ext

# for name in "${dataset[@]}"
# do
#     echo "Generating $name"
#     python3 d2ud.py ${path}/${name}/${name}.e
# done