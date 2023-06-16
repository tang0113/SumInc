cmd="mpirun -n 1 /home/tjn/SumInc-count_active_edge2/build/ingress -application pagerank -vfile /home/tjn/test/dataset/large/test/0.0000/test1.v -efile /home/tjn/test/dataset/large/test/sina/sina.edges  -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 1 -compress=1 -portion=1 -min_node_num=1 -max_node_num=5000 -sssp_source=0 -compress_concurrency=1 -compress_type=2 -serialization_prefix /home/tjn/test/ser/one -out_prefix /home/tjn/test/result/Ingress_pagerank -serialization_cmp_prefix /home/tjn/test/ser/two -message_type=push"
echo $cmd
eval $cmd
