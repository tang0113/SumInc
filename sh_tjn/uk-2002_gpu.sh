cmd="mpirun -n 1 /home/tjn/tjnCode/SumInc-count_active_edge2/build/ingress -application pagerank -vfile /home/tjn/test/dataset/large/test/0.0000/test1.v -efile /home/tjn/test/dataset/large/test/uk-2002/uk-2002.e  -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 1 -compress=0 -portion=1 -sssp_source=0 -serialization_prefix /home/tjn/test/ser/one -out_prefix /home/tjn/test/result/Ingress_pagerank gpu_start=1"
echo $cmd
eval $cmd
