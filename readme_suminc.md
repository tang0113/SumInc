# SumInc运行
## Suminc工作流程:
    Louvain压缩 生成聚类文件，标记每个点属于哪一个类别；
    # 2: 0 1
    # 3: 2 3 4
    Suminc 进行计算
    先过滤，生成超点；
    判断同一个类别的点当作一个超点是否满足索引的数量少于内部边的数量，满足则当作超点；
    为超点生成索引；
    利用索引计算；
## 构建相关文件：
    为了运行简单，请构建一个目录(root_path),在其下建立code、dataset、ser、result文件夹，dataset下面建立large、louvain_bin，层次关系如下：
    ├── code
    │   ├── gemini-graph
    │   ├── graphbolt
    │   ├── Ingress
    │   ├── libgrape-lite
    │   ├── louvain
    │   ├── SumInc
    │   └── Test_tools
    ├── dataset
    │   ├── large
    │   ├── louvain_bin
    │   └── ser
    ├── result
    │   ├── Ingress_pagerank
    │   └── sum_pagerank
    └── ser
## 构建测试数据集
    在large文件夹下，运行下面的代码建立test/0.0000文件夹
    mkdir test
    cd test
    mkdir 0.0000
      在/dataset/large/test下建立文件test.base, 复制内容如下：
    0 1
    0 2
    0 3
    1 4
    1 5
    1 2
    2 5
    2 3
    3 2
    3 6
    4 7
    5 7
    5 8
    5 9
    6 2
    6 8
    7 4
    8 9
    9 11
    10 9
    11 14
    14 12
    14 13
    14 15
    12 10
    12 13
    13 15
    9 16
    16 17
    16 18
    16 19
    17 20
    17 21
    18 21
    21 18
    19 21
    19 22
    20 21
    21 23
    22 23
    22 24
    23 24
    24 25
    24 26
    25 27
    26 25
    26 27
    27 25
    28 27
    29 9
    30 16
    31 23
    最后如下：
    ├── test
    │   └── 0.0000
    │       ├── test.base
## 运行压缩代码
    # 将下面的文件中root_path修改成自己路径
    ./code/louvain/gen-louvain/louvain.sh 
    运行Suminc
    # 编译方法
    ```
      mkdir build
      cd build
      cmake .. -DUSE_CILK=true -DCOUNT_AE=true # COUNT_AE表示开启活跃边统计
      make ingress
    ```
    # 将下面的文件中root_path修改成自己路径
    ./code/SumInc/sh/run_pr.sh
    ```
    mpirun -n 1 /mnt2/neu/yusong/code/SumInc_mirror_expr/SumInc_count_active_edge/build/ingress -application pagerank -vfile /mnt2/neu/yusong/dataset/expr_data_5/europe_osm/europe_osm.v -efile /mnt2/neu/yusong/dataset/expr_data_5/europe_osm/0.0001/europe_osm.base -efile_update /mnt2/neu/yusong/dataset/expr_data_5/europe_osm/0.0001/europe_osm.update -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -portion=1 -sssp_source=0 -php_source=0 -serialization_prefix /mnt2/neu/yusong/ser/suminc -verify=0

    mpirun -n 1 /mnt2/neu/yusong/code/SumInc_mirror_expr/SumInc_count_active_edge/build/ingress -application sssp -vfile /mnt2/neu/yusong/dataset/expr_data_5/test8/test8.v -efile /mnt2/neu/yusong/dataset/large/test8/0.0001/test8_w.base -efile_update /mnt2/neu/yusong/dataset/large/test8/0.0001/test8_w.update -directed=1 -cilk=true -termcheck_threshold 0.000001 -app_concurrency 16 -compress=1 -portion=0 -min_node_num=2 -max_node_num=5000 -sssp_source=0 -php_source=0 -compress_concurrency=1 -build_index_concurrency=16 -compress_type=2 -compress_threshold=1 -message_type=push -verify=0 -mirror_k=100000000 ma-count_skeleton=true --out_prefix ./
    ```

## 相关参数记录
  - portion=1: 表示开启优先级，否则关系这个仅仅对Iter类有用;
  - mirror_k: 表示建立Mirror时的阈值，特别的如果mirror_k=1e8时，关闭Mirror功能;
  - max_node_num: 指定cluster大小的上限，min_node_num为下限;

