#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <ctime>
#include <string>
#include <stdlib.h>
#include <cmath>
using namespace std;
using vid_t = int64_t;

void gen_vfile(std::string cfile, std::string vfile, vid_t node_num) {
  std::ifstream inFile(cfile);
  if(!inFile){
      std::cout << "open file failed. " << cfile << std::endl;
      exit(0);
  }

  std::cout << "reading cfile: " << cfile << std::endl;

  std::ofstream fout(vfile);
  if(!fout){
      std::cout << "open file failed. " << vfile << std::endl;
      exit(0);
  }

  std::vector<char> nodes;
  nodes.resize(node_num, 0);

  vid_t v_id = 0;
  vid_t size = 0;
  while(inFile >> size){
    for(int i = 0; i < size; i++){
      inFile >> v_id;
      if (v_id >= node_num) {
        std::cout << " error: vid >= node_num: " << v_id << " " << node_num 
                  << "\n";
        exit(0);
      }
      fout << v_id << std::endl;
      nodes[v_id] = 1;
    }
  }

  for (vid_t i = 0; i < node_num; i++) {
    if (nodes[i] == 0) {
      fout << i << std::endl;
    }
  }
  std::cout << "finish write vfile to " << vfile << std::endl;
}

int main(int argc, char **argv){
    std::cout << " Gen vfile..." << std::endl;
    // cmd: g++ gen_v_file.cc -o gen_v_file && ./gen_v_file cfile vfile node_num
    // eq: g++ gen_v_file.cc -o gen_v_file && ./gen_v_file /mnt2/neu/yusong/dataset/large/test8/0.0001/test8.base.c_5000 /mnt2/neu/yusong/dataset/large/test8/0.0001/test8.v 26

    std::string cluster_path = argv[1];
    std::string save_v_path = argv[2];
    vid_t node_size = std::stoi(argv[3]);;
    std::cout << "cluster_path=" << cluster_path << std::endl;
    std::cout << "write v_path=" << save_v_path << std::endl;
    std::cout << "node_size=" << node_size << std::endl;
    
    gen_vfile(cluster_path, save_v_path, node_size); // 读聚类的数据,并写入点数据

    std::cout << "- finish!!!\n" << std::endl;
    return 0;
}