cd /dataset
for dataset_name in google uk-2002
do 
    for i in 0 1 2
    do
        cmd="dataset_name=${dataset_name} i=${i}"
        echo $cmd
    done
done