/*
  处理数据集
    - 文本文件与二进制文件相互转换
    - 有向图转为无向图
    - 生成增量数据集
*/

#include <iostream>
#include <fstream>
#include<iomanip>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <cstdlib>
#include <time.h>
#include <string>
#include <chrono>
using namespace std;
using vid_t = uint64_t;
using w_t = uint64_t;
using edge_num_t = long long;

struct Edge
{
  vid_t to_vid;
  w_t weight=0;
  
  Edge(){}

  Edge(vid_t to, w_t w) {
    to_vid = to;
    weight = w;
  }

  bool operator==(const Edge b) const {  
    return this->to_vid == b.to_vid && this->to_vid == b.to_vid;  
  }
  bool operator<(const Edge b) const {  
    return this->to_vid < b.to_vid;  
  }
};

struct Edge_two{
  vid_t src_vid;
  vid_t to_vid;
  w_t weight=0;

  Edge_two(){}

  Edge_two(vid_t src, vid_t to, w_t w) {
    src_vid = src;
    to_vid = to;
    weight = w;
  }
};

DEFINE_string(base_edge, "", "efile path");
DEFINE_string(out_edge, "", "output efile path");
DEFINE_string(out_dir, "", "output efile dir");
DEFINE_bool(weight, false, "Whether the graph has the weight");
DEFINE_bool(hava_inadj, false, "Whether save in-adj of the graph");
DEFINE_bool(directed, false, "Whether is directed graph");
DEFINE_int32(head, 0, "file head");
DEFINE_double(inc_rate, 0, "rate of gen inc file");

auto formatDobleValue(double val, int fixed) {
    auto str = std::to_string(val);
    return str.substr(0, str.find(".") + fixed + 1);
}

class DealGraph{
  public:
  DealGraph() {

  }

  /**
   * 输出图的边, 只输出前100行
  */
  void print_graph() {
    LOG(INFO) << "---------------print graph----------------------";
    for (vid_t i = 0; i < this->node_num; i++) {
      for (auto e : this->graph[i]) {
        LOG(INFO) << "o_adj: " << i << "->" << e.to_vid << " " << e.weight;
      }
      if (FLAGS_hava_inadj) {
        for (auto e : this->graph_inadj[i]) {
          LOG(INFO) << "i_adj: " << i << "->" << e.to_vid << " " << e.weight;
        }
      }
      if (i > 10) {
        LOG(INFO) << " stop print: just print i<10.";
        break;
      }
    }
  }

  /**
   * 读文本文件的图
  */
  void read_graph_by_txt (const std::string iefile) {
    std::ifstream srcFile( iefile, ios::in );
    if ( !srcFile ) {
      LOG(INFO) << "error opening source: " << iefile;
      exit(0);
    }
    LOG(INFO) << "start read_graph_by_txt: " << iefile;

    char c[10000];
    for (int i = 0; i < FLAGS_head; i++) {
      srcFile.getline(c, 10000);
      LOG(INFO) << "ignore: " << c;
    }

    std::vector<std::pair<vid_t, Edge>> temp_graph;
    vid_t max_vid = 0;
    edge_num_t edge_num = 0;
    vid_t src_id = 0;
    vid_t to_id = 0;
    while ( srcFile >> src_id ) {
      Edge e;
      srcFile >> e.to_vid;
      max_vid = std::max(max_vid, src_id);
      max_vid = std::max(max_vid, e.to_vid);
      if (FLAGS_weight) {
        srcFile >> e.weight;
      }
      temp_graph.emplace_back(std::pair<vid_t, Edge>(src_id, e));
    }
    srcFile.close();

    this->node_num = max_vid;
    LOG(INFO) << " node_num=" << max_vid;
    LOG(INFO) << " edge_num=" << temp_graph.size();
    graph.resize(max_vid+1);
    if (FLAGS_hava_inadj) {
      graph_inadj.resize(max_vid+1);
    }
    for (auto p : temp_graph) {
      auto e = p.second;
      // LOG(INFO) << p.first << "->" << e.to_vid << " " << e.weight;
      graph[p.first].emplace_back(e);
      if (FLAGS_hava_inadj) {
        graph_inadj[e.to_vid].emplace_back(Edge(p.first, e.weight));
      }
    }
    LOG(INFO) << "  finish.";
  }

  /**
   * 读二进制文件的图
  */
  void read_graph_by_binary (const std::string iefile) {
    std::ifstream srcFile(iefile, ios::in | ios::binary);
    if ( !srcFile ) {
      LOG(INFO) << "error opening source: " << iefile;
      exit(0);
    }
    LOG(INFO) << "start read_graph_by_binary: " << iefile;

    std::vector<std::pair<vid_t, Edge>> temp_graph;
    vid_t max_vid = 0;
    edge_num_t edge_num = 0;
    vid_t src_id = 0;
    vid_t to_id = 0;
    while (srcFile.read( (char *) &src_id, sizeof(src_id))) {
      Edge e;
      srcFile.read((char*)&e.to_vid, sizeof(e.to_vid));
      max_vid = std::max(max_vid, src_id);
      max_vid = std::max(max_vid, e.to_vid);
      if (FLAGS_weight) {
        srcFile.read((char*)&e.weight, sizeof(e.weight));
      }
      // LOG(INFO) << src_id << " " << e.to_vid << ": " << e.weight;
      temp_graph.emplace_back(std::pair<vid_t, Edge>(src_id, e));
    }
    srcFile.close();

    this->node_num = max_vid;
    LOG(INFO) << " node_num=" << max_vid;
    LOG(INFO) << " edge_num=" << temp_graph.size();
    graph.resize(max_vid+1);
    if (FLAGS_hava_inadj) {
      graph_inadj.resize(max_vid+1);
    }
    for (auto p : temp_graph) {
      auto e = p.second;
      // LOG(INFO) << p.first << "->" << e.to_vid << " " << e.weight;
      graph[p.first].emplace_back(e);
      if (FLAGS_hava_inadj) {
        graph_inadj[e.to_vid].emplace_back(Edge(p.first, e.weight));
      }
    }
    LOG(INFO) << "  finish.";
  }

  /**
   * 将图写入二进制文件
  */
  void write_graph_to_binary (const std::string out_edge) {
    LOG(INFO) << "start write_graph_to_binary.";
    std::ofstream desFile( out_edge, ios::out | ios::binary );
    if ( !desFile ) {
      cout << " error opening destination file. " << out_edge << std::endl;
      desFile.close();
      return;
    }
    LOG(INFO) << "  start write_graph_to_binary: " << out_edge;

    for (vid_t i = 0; i < this->node_num; i++) {
      for (auto e : this->graph[i]) {
        // LOG(INFO) << i << "->" << e.to_vid << " " << e.weight;
        desFile.write((char *)&i, sizeof(i));
        desFile.write((char *)&e.to_vid, sizeof(e.to_vid));
        if (FLAGS_weight) {
          desFile.write((char *)&e.weight, sizeof(e.weight));
        }
      }
    }
    desFile.close();
    LOG(INFO) << "finish write_graph_to_binary.";
  }

  /**
   * 将图写入文本文件
  */
  void write_graph_to_txt (const std::string out_edge, 
                           const std::string separator=" ") {
    LOG(INFO) << "start write_graph_to_txt: " << out_edge;
    LOG(INFO) << "  separator=" << separator;
    std::ofstream desFile(out_edge);
    if ( !desFile ) {
      cout << "error opening destination file. " << out_edge << std::endl;
      desFile.close();
      return;
    }

    for (vid_t i = 0; i < this->node_num; i++) {
      for (auto e : this->graph[i]) {
        desFile << i << separator;
        desFile << e.to_vid;
        if (FLAGS_weight) {
          desFile << separator << e.weight;
        }
        desFile << "\n";
      }
    }
    desFile.close();
    LOG(INFO) << "  finish.";
  }

  /**
   * 有向图转无向图: 可能有重复边
  */
  void directedG_to_undirectedG () {
    LOG(INFO) << "directedG_to_undirectedG...";
    std::vector<std::vector<Edge> > graph_ud;
    graph_ud = this->graph;
    for (vid_t i = 0; i < this->node_num; i++) {
      for (auto e : this->graph[i]) {
        Edge r_e = e;
        r_e.to_vid = i;
        graph_ud[e.to_vid].emplace_back(r_e);
      }
    }
    this->graph = graph_ud;
    LOG(INFO) << "  graph -> graph_ud, finish.";
  }

  /**
   * 为边添加权重
  */
  void add_weight () {
    LOG(INFO) << "add_weight...";

    FLAGS_weight = true;
    // srand((unsigned)time(NULL));  // 关闭,便于调试
    for (vid_t i = 0; i < this->node_num; i++) {
      for (auto& e : this->graph[i]) {
        w_t w = rand() % 64 + 0; //生成[0, 64)范围内的随机数
        e.weight = w;
      }
    }
    LOG(INFO) << "  add_weight finish.";
  }

  /**
   * 根据增删比例生成增量需要的数据集
  */
  void gen_inc_data() {
    LOG(INFO) << "add_weight...";
    float inc_rate = FLAGS_inc_rate;
    std::string base_name = FLAGS_out_edge;
    std::string prefix = FLAGS_out_dir;
    LOG(INFO) << "  inc_rate=" << inc_rate;

    std::string command = "mkdir -p " + prefix + 
                          formatDobleValue(inc_rate, 4); // 生成4位的小数
    system(command.c_str());
    LOG(INFO) << " command=" << command;

    edge_num_t edge_num = graph.size();
    edge_num_t add_size, del_size;
    if (inc_rate >= 1) { // >=1, 这inc_rate表示需要增删的数量
      add_size = inc_rate / 2;
      del_size = inc_rate / 2;
    } else { // 删除指定比例
      add_size = edge_num * inc_rate / 2;
      del_size = edge_num * inc_rate / 2;
    }
    LOG(INFO) << "  edge_num=" << edge_num;
    LOG(INFO) << "  add_size=" << add_size;
    LOG(INFO) << "  del_size=" << del_size;

    float add_del_rate = 0.1;

    // +e: 2,3,-4, -e: 1,2,4,-4, e: 1,3,4
    std::ofstream fp_base(base_name + ".base");
    std::ofstream fp_update(base_name + ".update");           // for inc
    std::ofstream fp_updated(base_name + ".updated");         // for rerun
    std::ofstream fp_base_update(base_name + ".base_update"); // for risgraph
    LOG(INFO) << " save file: " << (base_name + ".base");

    std::vector<Edge_two> update_edges_del;
    std::vector<Edge_two> update_edges_add;

    size_t degree_eq0 = 0;
    for (vid_t i = 0; i < this->node_num; i++) {
      int i_degree = this->graph[i].size();
      // Compensate 0-degree
      if (i_degree == 0) {
        degree_eq0++;
        auto ies = this->graph_inadj[i];
        int rd = 0;
        if (ies.size() == 0) {
          rd = rand() % this->node_num + 0; // 生成[0, )范围内的随机数
          this->graph[i].emplace_back(rd, i + rd);
        } else {
          rd = rand() % ies.size() + 0;     // 生成[0, )范围内的随机数
          this->graph[i].emplace_back(Edge(ies[rd].to_vid, ies[rd].weight));
        }
      }
      for (auto e : this->graph[i]) {
        int rd = rand() % 100 + 0; // 生成[0, 100)范围内的随机数
        bool is_selected = false;
        // try delete
        if (rd == 0) {
          if (del_size > 0) {
            if (i_degree > 1) {
              // delete this edge
              is_selected = true;
              del_size--;
              i_degree--;
              // 1,2,4
              if (FLAGS_weight) {
                fp_base << i << " " << e.to_vid << " " 
                        << (i + e.to_vid)%128 << "\n";
                fp_update << "d " << i << " " << e.to_vid << " " 
                        << (i + e.to_vid)%128 << "\n";
              } else {
                fp_base << i << " " << e.to_vid << "\n";
                fp_update << "d " << i << " " << e.to_vid << "\n";
              }
              fp_base_update << i << " " << e.to_vid << "\n"; // 一定不带权
              update_edges_del.emplace_back(Edge_two(i, e.to_vid, e.weight));
            }
          }
        }
        // try delete, will not be selected at the same time as delete edge.
        if (rd == 1) {
          if (add_size > 0) {
            if (i_degree > 1) {
              // add this edge
              is_selected = true;
              add_size--;
              i_degree--;
              // 2,3,4
              if (FLAGS_weight) {
                fp_update << "a " << i << " " << e.to_vid << " " 
                          << (i + e.to_vid)%128 << "\n";
                fp_updated << i << " " << e.to_vid << " " 
                        << (i + e.to_vid)%128 << "\n";
              } else {
                fp_update << "a " << i << " " << e.to_vid << "\n";
                fp_updated << i << " " << e.to_vid << "\n";
              }
              update_edges_add.emplace_back(Edge_two(i, e.to_vid, e.weight));
            }
          }
        }
        // directly save to file
        if (is_selected == false) {
          // 1,3,4
          if (FLAGS_weight) {
            fp_base << i << " " << e.to_vid << " " << (i + e.to_vid)%128 << "\n";
            fp_updated << i << " " << e.to_vid << " " 
                    << (i + e.to_vid)%128 << "\n";
          } else {
            fp_base << i << " " << e.to_vid << "\n";
            fp_updated << i << " " << e.to_vid << "\n";
          }
          fp_base_update << i << " " << e.to_vid << "\n"; // 一定不带权
        }
      }
    }

    for (auto e : update_edges_del) {
      if (FLAGS_weight) {
        fp_base_update << e.src_vid << " " << e.to_vid << "\n";
      } else {
        fp_base_update << e.src_vid << " " << e.to_vid << "\n";
      }
    }
    for (auto e : update_edges_add) {
      if (FLAGS_weight) {
        fp_base_update << e.src_vid << " " << e.to_vid << "\n";
      } else {
        fp_base_update << e.src_vid << " " << e.to_vid << "\n";
      }
    }

    LOG(INFO) << "  degree_eq0=" << degree_eq0;
    LOG(INFO) << "  remaining add_size=" << add_size;
    LOG(INFO) << "  remaining del_size=" << del_size;
    LOG(INFO) << "  real update_edges_add.size=" << update_edges_add.size();
    LOG(INFO) << "  real update_edges_del.size=" << update_edges_del.size();

    // 记录删边/加边的数量，且这些边都位于文件末尾
    std::ofstream fp_base_update_num(base_name + ".base_update_num");
    fp_base_update_num << update_edges_del.size() << "\n";
    fp_base_update_num << update_edges_add.size() << "\n";
    fp_base_update_num << "del_num, add_num\n";
    fp_base_update_num.close();


    fp_base.close();
    fp_update.close();
    fp_updated.close();
    fp_base_update.close();
    LOG(INFO) << "gen_inc_data finish.";
  }

  /**
   * 根据增删比例生成增量需要的数据集
   *  生成带权和不带权的两种文件
  */
  void gen_inc_data_weight() {
    LOG(INFO) << "gen_inc_data_weight...";
    float inc_rate = FLAGS_inc_rate;
    std::string base_name = FLAGS_out_edge;
    std::string prefix = FLAGS_out_dir;
    LOG(INFO) << "  inc_rate=" << inc_rate;

    if (inc_rate > 1) {
      std::string command = "mkdir -p " + prefix + 
                            std::to_string(int(inc_rate)); // 生成4位的小数
      system(command.c_str());
      LOG(INFO) << "  command=" << command;
    } else {
      std::string command = "mkdir -p " + prefix + 
                            formatDobleValue(inc_rate, 4); // 生成4位的小数
      system(command.c_str());
      LOG(INFO) << "  command=" << command;
    }

    edge_num_t edge_num = graph.size();
    edge_num_t add_size, del_size;
    if (inc_rate > 1) { // if >1, inc_rate表示需要增删的数量，而不是比例
      add_size = inc_rate / 2;
      del_size = inc_rate / 2;
    } else { // 删除指定比例
      add_size = edge_num * inc_rate / 2;
      del_size = edge_num * inc_rate / 2;
    }
    LOG(INFO) << "  edge_num=" << edge_num;
    LOG(INFO) << "  add_size=" << add_size;
    LOG(INFO) << "  del_size=" << del_size;

    float add_del_rate = 0.1;

    // +e: 2,3,-4, -e: 1,2,4,-4, e: 1,3,4
    std::ofstream fp_base(base_name + ".base");
    std::ofstream fp_update(base_name + ".update");           // for inc
    std::ofstream fp_updated(base_name + ".updated");         // for rerun
    std::ofstream fp_base_w(base_name + "_w.base");
    std::ofstream fp_update_w(base_name + "_w.update");       // for inc
    std::ofstream fp_updated_w(base_name + "_w.updated");     // for rerun
    std::ofstream fp_base_update(base_name + ".base_update"); // for risgraph
    LOG(INFO) << " save file: " << (base_name + ".base");

    std::vector<Edge_two> update_edges_del;
    std::vector<Edge_two> update_edges_add;

    size_t degree_eq0 = 0;
    for (vid_t i = 0; i < this->node_num; i++) {
      int i_degree = this->graph[i].size();
      // Compensate 0-degree
      if (i_degree == 0) {
        degree_eq0++;
        auto ies = this->graph_inadj[i];
        int rd = 0;
        if (ies.size() == 0) {
          rd = rand() % this->node_num + 0; // 生成[0, )范围内的随机数
          this->graph[i].emplace_back(rd, (i + rd)%128);
        } else {
          rd = rand() % ies.size() + 0;     // 生成[0, )范围内的随机数
          this->graph[i].emplace_back(Edge(ies[rd].to_vid, ies[rd].weight));
        }
      }
      for (auto e : this->graph[i]) {
        int rd = rand() % 5 + 0; // 生成[0, 5)范围内的随机数
        bool is_selected = false;
        // try delete
        if (rd == 0) {
          if (del_size > 0) {
            if (i_degree > 3) {
              // delete this edge
              is_selected = true;
              del_size--;
              i_degree--;
              vid_t to_vid = e.to_vid;
              vid_t weight = (i + to_vid)%128;

              // for (auto e : this->graph[i])

              // 1,2,4
              {
                fp_base_w << i << " " << to_vid << " " 
                          << weight << "\n";
                fp_update_w << "d " << i << " " << to_vid << " " 
                            << weight << "\n";
              }
              {
                fp_base << i << " " << to_vid << "\n";
                fp_base_update << i << " " << to_vid << "\n";
                fp_update << "d " << i << " " << to_vid << "\n";
              }
              update_edges_del.emplace_back(Edge_two(i, to_vid, weight));
            }
          }
        }
        // try add, will not be selected at the same time as delete edge.
        if (rd == 1) {
          if (add_size > 0) {
            if (i_degree > 1) {
              // add this edge
              is_selected = true;
              add_size--;
              i_degree--;
              // 2,3,4
              {
                fp_update_w << "a " << i << " " << e.to_vid << " " 
                          << (i + e.to_vid)%128 << "\n";
                fp_updated_w << i << " " << e.to_vid << " " 
                        << (i + e.to_vid)%128 << "\n";
              }
              {
                fp_update << "a " << i << " " << e.to_vid << "\n";
                fp_updated << i << " " << e.to_vid << "\n";
              }
              update_edges_add.emplace_back(Edge_two(i, e.to_vid, e.weight));
            }
          }
        }
        // directly save to file
        if (is_selected == false) {
          // 1,3,4
          {
            fp_base_w << i << " " << e.to_vid << " " << (i + e.to_vid)%128 << "\n";
            fp_updated_w << i << " " << e.to_vid << " " 
                         << (i + e.to_vid)%128 << "\n";
          }
          {
            fp_base << i << " " << e.to_vid << "\n";
            fp_updated << i << " " << e.to_vid << "\n";
            fp_base_update << i << " " << e.to_vid << "\n";
          }
        }
      }
    }

    for (auto e : update_edges_del) {
      fp_base_update << e.src_vid << " " << e.to_vid << "\n";
    }
    for (auto e : update_edges_add) {
      fp_base_update << e.src_vid << " " << e.to_vid << "\n";
    }

    LOG(INFO) << "  degree_eq0=" << degree_eq0;
    LOG(INFO) << "  remaining add_size=" << add_size;
    LOG(INFO) << "  remaining del_size=" << del_size;
    LOG(INFO) << "  real update_edges_add.size=" << update_edges_add.size();
    LOG(INFO) << "  real update_edges_del.size=" << update_edges_del.size();

    // 记录删边/加边的数量，且这些边都位于文件末尾
    std::ofstream fp_base_update_num(base_name + ".base_update_num");
    fp_base_update_num << update_edges_del.size() << "\n";
    fp_base_update_num << update_edges_add.size() << "\n";
    fp_base_update_num << "del_num, add_num\n";
    fp_base_update_num.close();


    fp_base.close();
    fp_update.close();
    fp_updated.close();
    fp_base_w.close();
    fp_update_w.close();
    fp_updated_w.close();
    fp_base_update.close();
    LOG(INFO) << "gen_inc_data finish.";
  }


  ~DealGraph(){
  }

  std::vector<std::vector<Edge> > graph;
  std::vector<std::vector<Edge> > graph_inadj;
  vid_t node_num=0;
};

int main(int argc,char **argv) {
  /* 
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/large/test8/test8_w.base -out_edge=/mnt2/neu/yusong/dataset/large/test8/test8_w.base.b
  */
  auto start = std::chrono::system_clock::now();
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true; //设置日志消息是否转到标准输出而不是日志文件, 即true：不写入文件，只在终端显示
  FLAGS_colorlogtostderr = true;  // Set log color

  std::string base_edge = FLAGS_base_edge;
  std::string out_edge = FLAGS_out_edge;
  LOG(INFO) << base_edge;

  DealGraph graph;

  /* 
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/large/test8/test8_w.base -out_edge=/mnt2/neu/yusong/dataset/large/test8/test8_w.base.b

      -weight: 数据集是否有权
      -out_edge: 保存的文件路径
      -head: 表示数据文件前几行为为注释.

  */

  /* read txt graph, write binary graph */
  // graph.read_graph_by_txt(base_edge);
  // graph.write_graph_to_binary(out_edge);
  // graph.print_graph();


  /*
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/large/twitter/twitter/out.twitter -out_edge=/mnt2/neu/yusong/dataset/large/twitter/twitter.e -head=1
      -head: 表示数据文件前几行为为注释.
  */
  /* 读二进制文件，写txt文件 */
  // graph.read_graph_by_binary(base_edge);
  // graph.read_graph_by_txt(base_edge);
  // graph.write_graph_to_txt(out_edge);
  // graph.write_graph_to_binary(out_edge+".b");
  // graph.print_graph();

  /* 
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/large/google/google.e -out_edge=/mnt2/neu/yusong/dataset/large/google/google_w.e -head=4
      -head: 表示数据文件前几行为为注释.
  */
  /* 将无权文件转成有权文件 */
  // graph.read_graph_by_txt(base_edge);
  // graph.add_weight();
  // graph.write_graph_to_txt(out_edge);
  // graph.print_graph();

  /* 
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/large/google/google.e -out_edge=/mnt2/neu/yusong/dataset/large/google/google_ud.e -head=4
      -head: 表示数据文件前几行为为注释.
  */
  /* 读有向图文件,转成无向图文本文件 */
  // graph.read_graph_by_txt(base_edge);
  // graph.directedG_to_undirectedG();
  // graph.write_graph_to_txt(out_edge);
  // graph.print_graph();

  /*
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/large/test8/test8.e -out_edge=/mnt2/neu/yusong/dataset/large/test8/0.4000/test -out_dir=/mnt2/neu/yusong/dataset/large/test8/ -head=0 -inc_rate=0.4 -hava_inadj=1
      -head: 表示数据文件前几行为为注释.
  */
  /* 读二进制文件，写txt文件 */
  graph.read_graph_by_binary(base_edge + ".e.b");
  // graph.read_graph_by_txt(base_edge);
  // graph.gen_inc_data();     // 根据flag决定是否生存权重
  graph.gen_inc_data_weight(); // 生成带权和不带权的两种
  // graph.write_graph_to_txt(out_edge);
  // graph.print_graph();
  // graph.directedG_to_undirectedG();
  // graph.write_graph_to_binary(out_edge + "_ud.e.b");

  /*
    g++ deal_dataset.cc -lgflags -lglog -o deal_dataset && ./deal_dataset -weight=0 -base_edge=/mnt2/neu/yusong/dataset/expr_data/uk-2002/uk-2002.e -out_edge=/mnt2/neu/yusong/dataset/expr_data/uk-2002/uk-2002.csv -head=0
      -head: 表示数据文件前几行为为注释.
  */
  /* 读二进制文件，写csv文件 */
  // graph.read_graph_by_binary(base_edge + ".b");
  // // graph.read_graph_by_txt(base_edge);
  // graph.write_graph_to_txt(out_edge, ",");


  google::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  auto end = std::chrono::system_clock::now();
  fprintf(stderr, "time: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast
                                <std::chrono::microseconds>(end-start).count());
  return 0;
}