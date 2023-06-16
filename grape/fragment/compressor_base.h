#ifndef GRAPE_FRAGMENT_COMPRESSOR_BASE_H_
#define GRAPE_FRAGMENT_COMPRESSOR_BASE_H_

#include "grape/graph/super_node.h"
#include "grape/utils/Queue.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "timer.h"
#include "flags.h"
#include <iomanip>
#include "omp.h"
// #include <metis.h>

#define NUM_THREADS omp_get_max_threads()

namespace grape {

template <typename APP_T, typename SUPERNODE_T>
class CompressorBase : public ParallelEngine{
    public:
    using fragment_t = typename APP_T::fragment_t;
    using value_t = typename APP_T::value_t;
    using delta_t = typename APP_T::delta_t;
    using vertex_t = typename APP_T::vertex_t;
    using vid_t = typename APP_T::vid_t;
    using supernode_t = SUPERNODE_T;
    using fc_t = int32_t;
    using nbr_t = typename fragment_t::nbr_t;
    using nbr_index_t = Nbr<vid_t, delta_t>;
    using adj_list_t = typename fragment_t::adj_list_t;
    // using adj_list_index_t = AdjList<vid_t, value_t>;
    using adj_list_index_t = AdjList<vid_t, delta_t>; // for inc-sssp

    CompressorBase(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

    void init(const CommSpec& comm_spec, const Communicator& communicator,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()){
        comm_spec_ = comm_spec;
        communicator_ = communicator;
        InitParallelEngine(pe_spec);
        /* init */
        vid_t nodes_num = graph_->GetVerticesNum();
        // Fc.resize(nodes_num, FC_default_value);
        Fc.Init(graph_->Vertices(), FC_default_value);
        // Fc_map.Init(graph_->InnerVertices(), ID_default_value);
        id2spids.Init(graph_->Vertices(), ID_default_value);
        supernodes = new supernode_t[nodes_num];
        vid2in_mirror_cluster_ids.resize(nodes_num);
        vid2in_mirror_mids.resize(nodes_num);
        vid2out_mirror_mids.resize(nodes_num);
        // out_mirror2spids.resize(nodes_num);
        shortcuts.resize(nodes_num);
        old_node_num = nodes_num;
        all_node_num = nodes_num;
    }

    void print(std::string pos=""){
        LOG(INFO) << "--------------------------------print in " << pos;
        LOG(INFO) << "supernodes_num=" << supernodes_num << " ids_num=" << supernode_ids.size();
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            if (spn.id.GetValue() < old_node_num) {
              std::cout << "source_oid=" << graph_->GetId(spn.id) 
                        << " vid=" << spn.id.GetValue() << std::endl;
            } else {
              std::cout << "source_mirror_vid=" << spn.id.GetValue() << std::endl;
              std::cout << "  to_master_oid=" 
                        << graph_->GetId(mirrorid2vid[spn.id])
                        << "  master_vid=" << mirrorid2vid[spn.id].GetValue() << std::endl;
            }
            std::cout << " Fc_map=" << Fc_map[spn.id] 
                      << " spid=" << i << std::endl;
            std::cout << " ids_id=" << id2spids[spn.id];
            std::cout << " size=" << supernode_ids[spn.ids].size() << std::endl;
            for(auto u : supernode_ids[spn.ids]){
                std::cout << graph_->GetId(u) << " ";
            }
            std::cout << "\nFc:" << std::endl;
            for(auto fc : Fc){
                std::cout << fc << " ";
            }
            std::cout << std::endl;
            // LOG(INFO) << "inner_value:" << std::endl;
            // for(auto edge : spn.inner_value){
            //     LOG(INFO) << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
            // }
            std::cout << "inner_delta: size=" << spn.inner_delta.size() << std::endl;
            for(auto edge : spn.inner_delta){
                std::cout << "gid:" << this->v2Oid(edge.first) 
                          << ": " << edge.second << std::endl;
            }
            std::cout << "bound_delta: size=" << spn.bound_delta.size() << std::endl;
            for(auto edge : spn.bound_delta){
                std::cout << "gid:" << this->v2Oid(edge.first) 
                          << ": " << edge.second << std::endl;
            }
            std::cout << "-----------------------------------------" << std::endl;
        }
    }

    void write_spnodes(const std::string &efile){
        std::ofstream outfile(efile);
        if(!outfile){
            LOG(INFO) << "open file failed. " << efile;
            exit(0);
        }
        // //debug
        // {
        //     for(vid_t i = 0; i < supernodes_num; i++){
        //         supernode_t& spn = supernodes[i];
        //         for(auto edge : spn.inner_value){
        //             outfile << std::setprecision(10) << graph_->GetId(edge.first) << " " << edge.second << std::endl;
        //         }
        //     }
        //     return;
        // }
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            outfile << "id=" << graph_->GetId(spn.id) << " ids=" << spn.ids << " size=" << supernode_ids[spn.ids].size() << std::endl;
            for(auto u : supernode_ids[spn.ids]){
                outfile << graph_->GetId(u) << " ";
            }
            outfile << std::endl;
            // outfile << "inner_value:" << spn.inner_value.size() << std::endl;
            // for(auto edge : spn.inner_value){
            //     outfile << std::setprecision(10) << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
            // }
            outfile << "inner_delta:" << spn.inner_delta.size() << std::endl;
            for(auto edge : spn.inner_delta){
                // outfile << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
                vertex_t u;
                CHECK(graph_->Gid2Vertex(edge.second.parent_gid, u));
                outfile << graph_->GetId(edge.first) << ": " << edge.second.value << " id=" << graph_->GetId(u) << std::endl;
            }
            outfile << "bound_delta:" << spn.bound_delta.size() << std::endl;
            for(auto edge : spn.bound_delta){
                // outfile << graph_->GetId(edge.first) << ": " << edge.second << std::endl;
                vertex_t u;
                CHECK(graph_->Gid2Vertex(edge.second.parent_gid, u));
                outfile << graph_->GetId(edge.first) << ": " << edge.second.value << " id=" << graph_->GetId(u) << std::endl;
            }
            outfile << "fc:" << Fc[spn.id].size() << std::endl;
            for(auto f : Fc[spn.id]){
                outfile << graph_->GetId(f) << ",";
            }
            outfile << std::endl;
            outfile << std::endl;
        }
        LOG(INFO) << "finish write_spnodes..." << efile;
    }

    /*
        super_node:
        supernodes_num:
        id: vertex_t, vid 
        ids: vid
        Fc:
        supernode_ids： vertex_t
    */
    void write_spnodes_binary(const std::string &spnodefile){
        std::fstream file(spnodefile, std::ios::out | std::ios::binary);
        if(!file){
            LOG(INFO) << "Error opening file. " << spnodefile;
            exit(0);
        }
        // write supernode
        file.write(reinterpret_cast<char *>(&supernodes_num), sizeof(vid_t));
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            vid_t id = spn.id.GetValue();
            file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
            file.write(reinterpret_cast<char *>(&spn.ids), sizeof(vid_t));
        }
        // write Fc & supernode_source
        vid_t size;
        fc_t id;
        for(auto fc : Fc){
            id = fc;
            file.write(reinterpret_cast<char *>(&id), sizeof(fc_t));
        }
        vid_t source_num = supernode_source.size();
        file.write(reinterpret_cast<char *>(&source_num), sizeof(vid_t));
        for(auto ids : supernode_source){
            size = ids.size();
            file.write(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            vid_t id;
            for(auto f : ids){
                id = f.GetValue();
                file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
            }
        }
        // write supernode_ids
        vid_t ids_num = supernode_ids.size();
        file.write(reinterpret_cast<char *>(&ids_num), sizeof(vid_t));
        for(auto ids : supernode_ids){
            size = ids.size();
            file.write(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            vid_t id;
            for(auto f : ids){
                id = f.GetValue();
                file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
            }
        }
        file.close ();
    }

    /*
        super_node:
        id: vertex_t, vid 
        ids: vid
        Fc: vertex
        Fc_map:
        supernode_ids: vertex
        id2spids:
    */
    bool read_spnodes_binary(const std::string &spnodefile){
        std::fstream file(spnodefile, std::ios::in | std::ios::binary);
        if(!file){
            return false;
        }
        // read supernode
        file.read(reinterpret_cast<char *>(&supernodes_num), sizeof(vid_t));
        for(vid_t i = 0; i < supernodes_num; i++){
            supernode_t& spn = supernodes[i];
            vid_t id;
            file.read(reinterpret_cast<char *>(&id), sizeof(vid_t));
            spn.id = vertex_t(id);
            file.read(reinterpret_cast<char *>(&spn.ids), sizeof(vid_t));
            Fc_map[spn.id] = i;
        }
        // read Fc & supernode_source
        Fc.Init(graph_->Vertices(), FC_default_value);
        fc_t id = 0;
        vid_t i = 0;
        for(auto v : graph_->Vertices()){
            file.read(reinterpret_cast<char *>(&id), sizeof(fc_t));
            Fc[vertex_t(i)] = id;
            i++;
        }
        vid_t source_num = 0;
        file.read(reinterpret_cast<char *>(&source_num), sizeof(vid_t));
        if(source_num > 0){
            supernode_source.resize(source_num);
        }
        for(auto& ids : supernode_source){
            vid_t size = 0;
            file.read(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            if(size > 0){
                ids.resize(size);
            }
            vid_t id = 0;
            for(vid_t i = 0; i < size; i++){
                file.read(reinterpret_cast<char *>(&id), sizeof(vid_t));
                ids[i] = vertex_t(id);
            }
        }
        // read supernode_ids
        vid_t ids_num = 0;
        file.read(reinterpret_cast<char *>(&ids_num), sizeof(vid_t));
        if(ids_num > 0){
            supernode_ids.resize(ids_num);
        }
        vid_t cnt = 0;
        for(auto& ids : supernode_ids){
            vid_t size = 0;
            file.read(reinterpret_cast<char *>(&size), sizeof(ids.size()));
            if(size > 0){
                ids.resize(size);
            }
            vid_t id = 0;
            for(vid_t i = 0; i < size; i++){
                file.read(reinterpret_cast<char *>(&id), sizeof(vid_t));
                ids[i] = vertex_t(id);
                id2spids[ids[i]] = cnt;
            }
            cnt++;
        }
        file.close ();
        return true;
    }

    void compress(){
        std::string prefix = "";
        LOG(INFO) << FLAGS_serialization_cmp_prefix;
        if(!FLAGS_serialization_cmp_prefix.empty()){
            /*
                filename: efile + vfile + worknum + 
            */
            std::string serialize_prefix = FLAGS_serialization_cmp_prefix;
            std::string digest = FLAGS_efile + FLAGS_vfile + std::to_string(comm_spec_.worker_num());
            digest += "_" + std::to_string(comm_spec_.worker_id())
                    + "_" + std::to_string(FLAGS_max_node_num)
                    + "_" + std::to_string(FLAGS_min_node_num)
                    + "_" + std::to_string(FLAGS_compress_concurrency)
                    + "_" + std::to_string(FLAGS_directed)
                    + "_" + std::to_string(FLAGS_compress_type)
                    + "_mirror_k" + std::to_string(FLAGS_mirror_k)
                    + "_cmpthreshold" + std::to_string(FLAGS_compress_threshold);

            std::replace(digest.begin(), digest.end(), '/', '_');
            prefix = serialize_prefix + "/" + digest;
            LOG(INFO) << prefix;
            // if(read_spnodes_binary(prefix)){
            //     LOG(INFO) << "Deserializing supernode from " << prefix;
            //     return ;
            // }
            /* 无权和有权图的压缩完全一致: 故共用同一套序列化文件 */
            // std::replace(prefix.begin(), prefix.end(), "_w", "");
            std::string key = "_w";
            std::size_t found = prefix.rfind(key);
            if (found!=std::string::npos) {
                prefix.replace (found, key.length(), "");
            }
            LOG(INFO) << "prefix: " << prefix;
        }

        LOG(INFO) << "start compress...";

        /* use metis */
        if(FLAGS_compress_type == 1){
            // int thread_num = NUM_THREADS;
            int thread_num = FLAGS_compress_concurrency;
            // idx_t nParts = 10000;
            // idx_t nParts = graph_->GetVerticesNum() < FLAGS_max_node_num ? 1 : (thread_num*1);
            float tolerance_rate = 1.05;
            double start_time = GetCurrentTime();
            // metis_cut(nParts, tolerance_rate);
            // LOG(INFO) << "compress finish nparts-" << nParts << " use_time=" << (GetCurrentTime()-start_time);
            // Simulate thread pool
            // vid_t shared_id = 0;
            // ForEach(nParts, [this, &shared_id, &nParts](int tid) {
            //     LOG(INFO) << "build spnode, tid=" << tid << " begin...";
            //     int i = 0, cnt = 0, step = 1;
            //     while(i < nParts){
            //         i = __sync_fetch_and_add(&shared_id, step);
            //         if(i < nParts){
            //             for(auto u : graph_->InnerVertices()){
            //                 if(Fc[u] != FC_default_value || graph_part[u.GetValue()] != i) continue;
            //                 parallel_part_find_multi_source_supernode(u, i);
            //                 if(u.GetValue() % 1000000 == 0){
            //                     LOG(INFO) << "tid=" << tid << " vid=" << u.GetValue() << " spnodes_num=" << supernodes_num << std::endl;
            //                 }
            //             }
            //             cnt++;
            //         }
            //     }
            //     LOG(INFO) << "tid=" << tid << " finish! cnt=" << cnt;
            //     }, thread_num
            // );
            // compress_by_metis(nParts);
            // if(!FLAGS_serialization_cmp_prefix.empty()){
            //     LOG(INFO) << "Serializing supernode to " << prefix;
            //     write_spnodes_binary(prefix);
            // }
        }
        /* use scan++/ Louvain */
        else if(FLAGS_compress_type == 2){
            // use scan++ cluster method
            // compress_by_scanpp();
            compress_by_cluster(prefix);
        }
        else{
            /* find multi-source supernode */
            auto inner_vertices = graph_->InnerVertices();
            if(FLAGS_compress_concurrency == -1){
                for(auto u : inner_vertices){
                    if(Fc[u] != FC_default_value) continue;
                    parallel_find_multi_source_supernode(u, 0, graph_->GetVerticesNum(), FLAGS_min_node_num);
                    if(u.GetValue() % 1000000 == 0){
                        LOG(INFO) << "----id=" << u.GetValue() << " spnodes_num=" << supernodes_num << std::endl;
                    }
                }
            }
            else{
                // sort by out degree
                vid_t nodes_num = graph_->GetVerticesNum();
                std::vector<std::pair<vertex_t, vid_t>> id_outdegree;
                id_outdegree.reserve(nodes_num);
                for(auto u : inner_vertices){
                    id_outdegree.emplace_back(std::pair<vertex_t, vid_t>(u, graph_->GetOutgoingAdjList(u).Size())); // out degree
                    // id_outdegree.emplace_back(std::pair<vertex_t, vid_t>(u, graph_->GetIncomingAdjList(u).Size())); // in degree
                }

                LOG(INFO) << "all_num=" << graph_->GetVerticesNum();
                int batch = 3;
                ForEachCompress(inner_vertices, [this, &inner_vertices, &batch, &id_outdegree](int tid, vid_t begin, vid_t end) {
                    LOG(INFO) << "tid=" << tid << " range: "<< begin << "->" << end;
                    // sort
                    std::sort(id_outdegree.begin() + begin, id_outdegree.begin() + end, cmp_pair_b2s); // big to small
                    // std::sort(id_outdegree.begin() + begin, id_outdegree.begin() + end, cmp_pair_s2b_sort); // small to big
                    
                    // for(int i = begin; i < end; i++){
                    //     auto& pair = id_outdegree[i];
                    //     LOG(INFO) << "tid=" << tid << " sorted: " << pair.first.GetValue() << ": " << pair.second;
                    // }
                    for(int i = 1; i <= 3; i++){
                        int b = FLAGS_min_node_num * i;
                        for(vid_t i = begin; i < end; i++){
                            // vertex_t u(i);
                            vertex_t u(id_outdegree[i].first);
                            if(Fc[u] != FC_default_value) continue;
                            // parallel_find_multi_source_supernode(u, begin, end, b);
                            // parallel_find_multi_source_supernode_2(u, begin, end, b);
                            // parallel_find_multi_source_supernode_2_sort(u, begin, end, b);
                            parallel_find_multi_source_supernode_2_sort_sample(u, begin, end, b);
                            if(i % 1000000 == 0){
                                LOG(INFO) << "tid=" << tid << " vid=" << u.GetValue() << " spnodes_num=" << supernodes_num << std::endl;
                            }
                        }
                        LOG(INFO) << "--tid=" << tid << " min_node_num=" << b;
                    }
                    LOG(INFO) << "tid=" << tid << " finish!";
                }, FLAGS_compress_concurrency
                );
                MPI_Barrier(comm_spec_.comm());
            }
            if(!FLAGS_serialization_cmp_prefix.empty()){
                LOG(INFO) << "Serializing supernode to " << prefix;
                write_spnodes_binary(prefix);
            }
        }


        LOG(INFO) << "finish compress...";
    }
    
    void statistic(){
        long inner_edges_num = 0;
        long bound_edges_num = 0;
        // long inner_value_num = 0;
        long supernodes_comtain_num = 0;
        long max_node_num = 0;
        long min_node_num = std::numeric_limits<vid_t>::max();
        long max_inner_edges_num = 0;
        long max_bound_edges_num = 0;
        for(long i = 0; i < supernodes_num; i++){
            inner_edges_num += supernodes[i].inner_delta.size();
            // inner_value_num += supernodes[i].inner_value.size();
            bound_edges_num += supernodes[i].bound_delta.size();
            max_node_num = std::max(max_node_num, (long)supernode_ids[supernodes[i].ids].size());
            min_node_num = std::min(min_node_num, (long)supernode_ids[supernodes[i].ids].size());
            max_inner_edges_num = std::max(max_inner_edges_num, (long)supernodes[i].inner_delta.size());
            max_bound_edges_num = std::max(max_bound_edges_num, (long)supernodes[i].bound_delta.size());
        }
        for(auto ids : supernode_ids){
            supernodes_comtain_num += ids.size();
        }
        long bound_node_num = 0;
        for(auto f : supernode_out_bound){
            if(f){
                bound_node_num++;
            }
        }
        long source_node_num = 0;
        for(auto vec : supernode_source){
            source_node_num += vec.size();
        }

        long global_bound_node_num = 0;
        long global_source_node_num = 0;
        long global_spn_com_num = 0;
        long global_spn_num = 0;
        long global_inner_edges_num = 0;
        // long global_inner_value_num = 0;
        long global_bound_edges_num = 0;
        long global_max_node_num = 0;
        long global_min_node_num = 0;
        long global_max_inner_edges_num = 0;
        long global_max_bound_edges_num = 0;
        long nodes_num = graph_->GetTotalVerticesNum();
        long edges_num = 0;
        long local_edges_num = graph_->GetEdgeNum();
        long local_ids_num = supernode_ids.size();
        long global_ids_num = 0;

        communicator_.template Sum(source_node_num, global_source_node_num);
        communicator_.template Sum(bound_node_num, global_bound_node_num);
        communicator_.template Sum(supernodes_comtain_num, global_spn_com_num);
        communicator_.template Sum((long)supernodes_num, global_spn_num);
        communicator_.template Sum(inner_edges_num, global_inner_edges_num);
        // communicator_.template Sum(inner_value_num, global_inner_value_num);
        communicator_.template Sum(bound_edges_num, global_bound_edges_num);
        communicator_.template Sum(local_edges_num, edges_num);
        communicator_.template Sum(local_ids_num, global_ids_num);
        communicator_.template Max(max_node_num, global_max_node_num);
        communicator_.template Max(min_node_num, global_min_node_num);
        communicator_.template Max(max_inner_edges_num, global_max_inner_edges_num);
        communicator_.template Max(max_bound_edges_num, global_max_bound_edges_num);

        edges_num /= 2; // in/out

        LOG(INFO) << "efile=" << FLAGS_efile;
        LOG(INFO) << "work" << comm_spec_.worker_id() << " supernodes_num=" << supernodes_num;
        if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "statistic...";
            LOG(INFO) << "#graph edges_num: " << edges_num;
            LOG(INFO) << "#nodes_num: " << nodes_num << std::endl;
            LOG(INFO) << "#global_spn_num: " << global_spn_num << std::endl; 
            LOG(INFO) << "#global_ids_num: " << global_ids_num << std::endl; 
            LOG(INFO) << "#global_spn_com_num: " << global_spn_com_num  << std::endl; 
            LOG(INFO) << "#global_spn_com_num_ave: " << (global_spn_com_num/(global_ids_num+1e-3))  << std::endl; 
            LOG(INFO) << "#global_bound_node_num: " << global_bound_node_num  << std::endl; 
            LOG(INFO) << "#global_source_node_num: " << global_source_node_num  << std::endl; 
            LOG(INFO) << "#global_inner_index_num: " << global_inner_edges_num  << std::endl; 
            LOG(INFO) << "#global_bound_index_num: " << global_bound_edges_num  << std::endl; 
            LOG(INFO) << "#global_spn_com_num/nodes_num: " << (global_spn_com_num*1.0/nodes_num)  << std::endl; 
            LOG(INFO) << "#supernodes_index/edges_num: " << ((global_inner_edges_num+global_bound_edges_num)*1.0/edges_num)  << std::endl; 
            // LOG(INFO) << "global_inner_value_num=" << global_inner_value_num << std::endl;
            LOG(INFO) << "#max_node_num: " << global_max_node_num << std::endl; 
            LOG(INFO) << "#min_node_num: " << global_min_node_num << std::endl; 
            LOG(INFO) << "#max_inner_index_num: " << global_max_inner_edges_num << std::endl; 
            LOG(INFO) << "#max_bound_index_num: " << global_max_bound_edges_num << std::endl; 
            LOG(INFO) << "#MAX_NODE_NUM: " << MAX_NODE_NUM  << std::endl; 
            LOG(INFO) << "#MIN_NODE_NUM: " << MIN_NODE_NUM  << std::endl; 
        }

        //debug
        {
            long long bound_edge_num = 0;
            long long inner_edge_num = 0;
            long long filter_num = 0;
            long long filter_error = 0;
            long long filter_save = 0;
            long long inner_edge_num_ = 0;
            long long best_save = 0;
            // check P set
            for(vid_t j = 0; j < supernode_ids.size(); j++){  // parallel compute
                std::vector<vertex_t> &node_set = this->supernode_ids[j];
                long long temp_ie_num = 0;
                vertex_t s;
                // inner edge, out bound edge
                for(auto v : node_set){
                    auto spids = this->id2spids[v];
                    const auto& oes = graph_->GetOutgoingAdjList(v);
                    for(auto& e : oes){
                        if(this->id2spids[e.neighbor] == spids){
                            inner_edge_num++;
                            temp_ie_num++;
                        }
                        else{
                            bound_edge_num++;
                        }
                    }
                    s = v;
                }
                // inner bound node
                int b_num = 0;
                for(auto v : node_set){
                    auto spids = this->id2spids[v];
                    const auto& oes = graph_->GetOutgoingAdjList(v);
                    for(auto& e : oes){
                        if(this->id2spids[e.neighbor] != spids){
                            b_num++;
                            break;
                        }
                    }
                }
                // source node
                int s_num_new = 0;
                for(auto v : node_set){
                    auto spids = this->id2spids[v];
                    const auto& oes = graph_->GetIncomingAdjList(v);
                    for(auto& e : oes){
                        if(this->id2spids[e.neighbor] != spids){
                            s_num_new++;
                            break;
                        }
                    }
                }
                vid_t src_id = Fc[s] < 0 ? (-Fc[s]-1) : Fc[s];
                int s_num = supernode_source[src_id].size();
                if(s_num * b_num > temp_ie_num && s_num > 1){
                    filter_num++;
                    if (filter_num < 5) {
                        LOG(INFO) << "id=" << j << " s_num=" << s_num << " b_num=" << b_num << " temp_ie_num=" << temp_ie_num;
                    }
                    // CHECK_EQ(s_num, s_num_new);
                }
            }

            LOG(INFO) << " filter_num=" << filter_num << " filter_error=" << filter_error << " filter_save=" << filter_save;
            LOG(INFO) << "#inner_edge_num/global_bound_edges_num: " << (inner_edge_num*1.0/global_bound_edges_num) << std::endl;
            LOG(INFO) << "#save_edge_rate: " << ((inner_edge_num*1.0-global_bound_edges_num)/edges_num) << std::endl;
            LOG(INFO) << "#inner_edge_num: " << inner_edge_num << std::endl; 
            // LOG(INFO) << "#bound_edge_num: " << bound_edge_num << std::endl; 
            LOG(INFO) << "#inner_edge_num/edges_num: " << (inner_edge_num*1.0/edges_num) << std::endl; 
            
            // check not filter
//             {
//                 for(vid_t i = 0; i < supernode_source.size(); i++){
//                     std::vector<vertex_t> &source_set = this->supernode_source[i];
//                     long long index_sum = 0;
//                     long long ie_sum = 0;
//                     long long s_num_ = 0;
//                     long long b_num_ = 0;
//                     long long node_num_ = 0;
//                     for(auto u : source_set){
//                         supernode_t &spnode = supernodes[Fc_map[u]];
//                         // supernode_t &spnode = supernodes[i];
//                         std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];
//                         long long temp_ie_num = 0;
//                         vertex_t s;
//                         // inner edge, out bound edge
//                         for(auto v : node_set){
//                             auto spids = this->id2spids[v];
//                             const auto& oes = graph_->GetOutgoingAdjList(v);
//                             for(auto& e : oes){
//                                 if(this->id2spids[e.neighbor] == spids){
//                                     inner_edge_num++;
//                                     temp_ie_num++;
//                                 }
//                                 else{
//                                     bound_edge_num++;
//                                 }
//                             }
//                             s = v;
//                         }
//                         // inner bound node
//                         int b_num = 0;
//                         for(auto v : node_set){
//                             auto spids = this->id2spids[v];
//                             const auto& oes = graph_->GetOutgoingAdjList(v);
//                             for(auto& e : oes){
//                                 if(this->id2spids[e.neighbor] != spids){
//                                     b_num++;
//                                     break;
//                                 }
//                             }
//                         }
//                         // source node
//                         int s_num_new = 0;
//                         for(auto v : node_set){
//                             auto spids = this->id2spids[v];
//                             const auto& oes = graph_->GetIncomingAdjList(v);
//                             for(auto& e : oes){
//                                 if(this->id2spids[e.neighbor] != spids){
//                                     s_num_new++;
//                                     break;
//                                 }
//                             }
//                         }
//                         vid_t src_id = Fc[s] < 0 ? (-Fc[s]-1) : Fc[s];
//                         int s_num = supernode_source[src_id].size();
//                         if(s_num * b_num > temp_ie_num && cnt < 20 && s_num > 1){
//                             cnt++;
//                             LOG(INFO) << "id=" << i << " s_num=" << s_num << " b_num=" << b_num << " temp_ie_num=" << temp_ie_num;
//                             CHECK_EQ(s_num, s_num_new);
//                         }
//                         ie_sum = temp_ie_num;
//                         s_num_ = s_num;
//                         b_num_ = b_num;
//                         node_num_ = node_set.size();
//                         index_sum += spnode.bound_delta.size();
//                     }
//                     inner_edge_num_ += ie_sum;
//                     if(s_num_ * b_num_ <= ie_sum){
//                         LOG(INFO) << "i="<< i << "---no --- " << " s_num=" << s_num_ << " b_num=" << b_num_ << " ie_num=" << ie_sum << " real_index_num=" << index_sum << " node_sum=" << node_num_;
//                         filter_num++;
//                         best_save += ie_sum - index_sum;
//                     }
//                     else{
//                         if(ie_sum >= index_sum){
//                             filter_error++;
//                             filter_save += ie_sum - index_sum;
//                             LOG(INFO) << "i="<< i << "---filter error- -- " << " s_num=" << s_num_ << " b_num=" << b_num_ << " ie_num=" << ie_sum << " real_index_num=" << index_sum << " node_sum=" << node_num_;
//                         }
//                         else{
//                             LOG(INFO) << "i="<< i << "---filter yes--- " << " s_num=" << s_num_ << " b_num=" << b_num_ << " ie_num=" << ie_sum << " real_index_num=" << index_sum << " node_sum=" << node_num_;
//                         }
//                     }
//                 }
                // LOG(INFO) << "filte_save=" << best_save << " best_save=" << (best_save+filter_save);
                // LOG(INFO) << "filte_save_rate=" << (best_save*1.0/edges_num);
                // LOG(INFO) << "best_save_rate=" << ((best_save+filter_save)*1.0/edges_num);
                // LOG(INFO) << " filter_num=" << filter_num << " filter_error=" << filter_error << " filter_save=" << filter_save;
                // LOG(INFO) << "#inner_edge_num/global_bound_edges_num: " << (inner_edge_num_*1.0/global_bound_edges_num) << std::endl;
                // LOG(INFO) << "#save_edge_rate: " << ((inner_edge_num_*1.0-global_bound_edges_num)/edges_num) << std::endl;
                // LOG(INFO) << "#inner_edge_num: " << inner_edge_num_ << std::endl; 
                // // LOG(INFO) << "#bound_edge_num: " << bound_edge_num << std::endl; 
                // LOG(INFO) << "#inner_edge_num/edges_num: " << (inner_edge_num_*1.0/edges_num) << std::endl; 
//          }
        }

        // Statistics superpoint size distribution data
        int step = 500;
        int max_num = 1e6;
        std::vector<int> s(max_num/step, 0);
        for (int i=0 ; i < supernode_ids.size(); i++) {
            s[supernode_ids[i].size()/step]++;
        }
        for(int i = 0; i < max_num/step; i++){
            if(s[i] > 0){
                LOG(INFO) << "   statistic: " << (i*step) << "-" << (i*step+step) << ":" << s[i];
            }
        }
    }

    // cost mode1: 
    void parallel_find_multi_source_supernode(const vertex_t source, vid_t begin, vid_t end, int now_min_node_num){
        // std::unordered_set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        // std::unordered_set<vertex_t> P; // inner node
        // std::unordered_set<vertex_t> O; // V^out
        std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        std::set<vertex_t> P; // inner node
        std::set<vertex_t> O; // V^out
        std::queue<vertex_t> D; // candidate queue
        // std::unordered_set<vertex_t> visited_nodes;
        std::set<vertex_t> visited_nodes;
        fid_t fid_ = graph_->GetFragId(source);
        // add source
        visited_nodes.insert(source);
        P.insert(source);
        const auto& ies = graph_->GetIncomingAdjList(source);
        for(auto& e : ies){
            if(e.neighbor != source){
                S.insert(source);
                break;
            }
        }
        const auto& oes = graph_->GetOutgoingAdjList(source);
        for(auto& e : oes){
            auto& u = e.neighbor;
            if(u.GetValue() >= begin && u.GetValue() < end && visited_nodes.find(u) == visited_nodes.end() && Fc[u] == FC_default_value && fid_ == graph_->GetFragId(u)){
                // D.push(u);
                // visited_nodes.insert(u);
                auto noe = graph_->GetOutgoingAdjList(u);
                if(noe.Size() <= MAX_NODE_NUM){ // filter powerlog node!!!
                    D.push(u);
                    visited_nodes.insert(u);
                }
            }
            if(O.find(u) == O.end() && P.find(u) == P.end()){
                O.insert(u);
            }
        }
        vid_t inner_edge_num = 0;
        vid_t bound_edge_num = O.size();
        vid_t temp_inner = 0;
        vid_t temp_bound = 0;
        float pre_score = 0; // pre compressed score
        int step = 0;
        while(!D.empty() && P.size() <= MAX_NODE_NUM){
            step++;
            // if(step > 1000){
            if(step > 5000){
                LOG(INFO) << "step>5000, workid=" << comm_spec_.worker_id() << " D.size=" << D.size() << " P.size=" << 
                P.size();
                // exit(0);
                break;
            }
            vertex_t d = D.front();
            D.pop();
            // if d add to P:
            bool is_s = false;
            temp_inner = inner_edge_num;
            temp_bound = bound_edge_num;
            for(auto& u : graph_->GetIncomingAdjList(d)){
                if(P.find(u.neighbor) != P.end()){
                    temp_bound--;
                    temp_inner++;
                }
                else{
                    is_s = true;
                }
            }
            for(auto& u : graph_->GetOutgoingAdjList(d)){
                if(P.find(u.neighbor) == P.end()){
                    temp_bound++;
                }
                else{
                    temp_inner++;
                }
            }
            // update S/P
            std::unordered_set<vertex_t> temp_O;
            std::unordered_set<vertex_t> temp_S;
            // temp_S.insert(S.begin(), S.end());
            vid_t now_s_size = S.size();
            vid_t now_o_size = O.size();
            for(auto& v : graph_->GetOutgoingAdjList(d)){
                if(S.find(v.neighbor) != S.end()){
                    bool flag = true;
                    for(auto& u : graph_->GetIncomingAdjList(v.neighbor)){
                        if(P.find(u.neighbor) == P.end() && u.neighbor != d){
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        // temp_S.erase(v.neighbor);
                        temp_S.insert(v.neighbor);
                        now_s_size--;
                    }
                }
            }
            if(is_s){
                // temp_S.insert(d);
                now_s_size++;
            }
            // temp_O.insert(O.begin(), O.end());
            // temp_O.erase(d);
            now_o_size--; // delete d
            for(auto& u : graph_->GetOutgoingAdjList(d)){
                if(P.find(u.neighbor) == P.end() && O.find(u.neighbor) == O.end() && u.neighbor != d){
                    temp_O.insert(u.neighbor);
                }
            }
            now_o_size += temp_O.size();

            // float now_score = temp_inner * 1.0 / (temp_bound + P.size());
            // float now_score = temp_inner * 1.0 / (temp_S.size() * temp_O.size());
            // if((P.size() < 4) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            // if((P.size() < MIN_NODE_NUM-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            // if((P.size() < now_min_node_num-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            if((P.size() < now_min_node_num-1) || (now_s_size * now_o_size < temp_inner * 1 + temp_bound)){
            // if((now_score >= pre_score && P.size() < now_min_node_num-1) || now_score >= 1){
                /* update O */
                // temp_O.swap(O);
                O.erase(d);
                O.insert(temp_O.begin(), temp_O.end());
                /* update S */
                // temp_S.swap(S);
                if(is_s){
                    S.insert(d);
                }
                for(auto v : temp_S){
                    S.erase(v);
                }
                // pre_score = now_score;
                inner_edge_num = temp_inner;
                bound_edge_num = temp_bound;
                P.insert(d);
                for(auto& u : graph_->GetOutgoingAdjList(d)){
                    if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
                        // D.push(u.neighbor);
                        // visited_nodes.insert(u.neighbor);
                        auto noe = graph_->GetOutgoingAdjList(u.neighbor);
                        if(noe.Size() <= MAX_NODE_NUM && D.size() <= MAX_NODE_NUM * 10){ // filter some nodes that hava big degree!!!
                            D.push(u.neighbor);
                            visited_nodes.insert(u.neighbor);
                        }
                    }
                }
            }
        }

        // if(P.size() >= MIN_NODE_NUM){
        if(P.size() >= now_min_node_num){
            // if(bound_edge_num > 500){
            //     LOG(INFO) << "--pre_score=" << pre_score;
            // }
            /* create a set of indexes for each entry */
            if(S.size() == 0){
                S.insert(source); // There is no entry point. At this time, the source point is used to form a super point.
            }
            int ids_id = -1;
            {
                std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                supernode_ids.emplace_back(P.begin(), P.end());
                ids_id = int(supernode_ids.size()) - 1;
                // supernode_bound_ids.emplace_back(O.begin(), O.end());
                supernode_source.emplace_back(S.begin(), S.end());
            }
            // CHECK(ids_id >= 0);
            for(auto u : P){
                Fc[u] = -(ids_id+1);
                id2spids[u] = ids_id;
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                // vid_t supernode_id = supernodes_num;
                // supernodes_num++;
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id;
            }
        }
    }

    /* new cost mode: mode2 */
    void parallel_find_multi_source_supernode_2(const vertex_t source, vid_t begin, vid_t end, int now_min_node_num){
        // std::unordered_set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        // std::unordered_set<vertex_t> P; // inner node
        // std::unordered_set<vertex_t> O; // V^out
        std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        std::set<vertex_t> P; // inner node
        std::set<vertex_t> B; // belong to P, bound vertices
        std::queue<vertex_t> D; // candidate queue
        // std::unordered_set<vertex_t> visited_nodes;
        std::set<vertex_t> visited_nodes;
        fid_t fid_ = graph_->GetFragId(source);
        const int try_max_node = MAX_NODE_NUM * 1000; // 1 776 852
        const int try_max_step = MAX_NODE_NUM * 1000;
        const float cost_ratio = 1;

        // add source
        /*
        visited_nodes.insert(source);
        P.insert(source);
        const auto& ies = graph_->GetIncomingAdjList(source);
        for(auto& ie : ies){
            if(ie.neighbor != source){
                S.insert(source);
                break;
            }
        }
        const auto& oes = graph_->GetOutgoingAdjList(source);
        bool flag = false;
        if(oes.Size() < try_max_node){
            for(auto& oe : oes){
                auto& u = oe.neighbor;
                if(u.GetValue() >= begin && u.GetValue() < end && visited_nodes.find(u) == visited_nodes.end() && Fc[u] == FC_default_value && fid_ == graph_->GetFragId(u)){
                    D.push(u);
                    visited_nodes.insert(u);
                }
                if(P.find(u) == P.end()){
                    flag = true;
                }
            }
        }
        if(flag){
            B.insert(source);
        }
        */
        if(source.GetValue() >= begin && source.GetValue() < end && Fc[source] == FC_default_value && fid_ == graph_->GetFragId(source)){
            D.push(source);
            visited_nodes.insert(source);
        }

        vid_t inner_edge_num = 0;
        vid_t temp_inner = 0;
        float pre_score = 0; // pre compressed score
        int step = 0;
        long long last_s_size = 0;
        long long last_b_size = 0;
        long long last_ie_num = 0;
        while(!D.empty() && P.size() <= MAX_NODE_NUM){
            step++;
            // if(step > 1000){
            if(step > try_max_step){
                LOG(INFO) << "step>" << try_max_step << ", workid=" << comm_spec_.worker_id() << " D.size=" << D.size() << " P.size=" << 
                P.size();
                break;
            }
            vertex_t d = D.front();
            D.pop();
            // if d add to P:
            bool is_s = false;
            bool is_b = false;
            temp_inner = inner_edge_num;
            // for(auto& ie : graph_->GetIncomingAdjList(d)){
            //     if(P.find(ie.neighbor) != P.end()){
            //         temp_inner++;
            //     }
            //     else{
            //         is_s = true;
            //     }
            // }
            // for(auto& oe : graph_->GetOutgoingAdjList(d)){
            //     if(!is_b && P.find(oe.neighbor) == P.end()){
            //         is_b = true;
            //     }
            //     else{
            //         temp_inner++;
            //     }
            // }
            // update S/P/O
            std::unordered_set<vertex_t> temp_B; // need to delete
            std::unordered_set<vertex_t> temp_S; // need to delete
            vid_t now_s_size = S.size();
            vid_t now_b_size = B.size();
            for(auto& oe : graph_->GetOutgoingAdjList(d)){
                // update S
                if(S.find(oe.neighbor) != S.end()){
                    bool flag = true;
                    for(auto& ie : graph_->GetIncomingAdjList(oe.neighbor)){
                        if(P.find(ie.neighbor) == P.end() && ie.neighbor != d){
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        // temp_S.erase(v.neighbor);
                        temp_S.insert(oe.neighbor);
                        // now_s_size--;
                    }
                }
                // update temp_inner
                if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){ // self-edge
                    is_b = true;
                }
                else{
                    temp_inner++;
                }
            }
            for(auto& ie : graph_->GetIncomingAdjList(d)){
                // update B
                if(B.find(ie.neighbor) != B.end()){
                    bool flag = true;
                    for(auto& oe : graph_->GetOutgoingAdjList(ie.neighbor)){
                        if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        temp_B.insert(ie.neighbor);
                        // now_b_size--;
                    }
                }
                // update temp_inner
                // if(P.find(ie.neighbor) != P.end() || ie.neighbor == d){ // self-edge
                if(P.find(ie.neighbor) != P.end()){ // self-edge just only count once, it has been calculated in the out side.
                    temp_inner++;
                }
                else{
                    is_s = true;
                }
            }
            now_s_size -= temp_S.size();
            now_b_size -= temp_B.size();
            if(is_s){
                // temp_S.insert(d);
                now_s_size++;
            }
            if(is_b){
                // B.insert(d);
                now_b_size++;
            }

            // float now_score = temp_inner * 1.0 / (temp_S.size() * temp_O.size());
            // if((P.size() < 4) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            // if((P.size() < MIN_NODE_NUM-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            // if((P.size() < now_min_node_num-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            if((P.size() < now_min_node_num-1) || (now_s_size * now_b_size < temp_inner * cost_ratio)){
            // if((now_score >= pre_score && P.size() < now_min_node_num-1) || now_score >= 1){
                /* update B */
                if(is_b){
                    B.insert(d);
                }
                for(auto v : temp_B){
                    B.erase(v);
                }
                /* update S */
                if(is_s){
                    S.insert(d);
                }
                for(auto v : temp_S){
                    S.erase(v);
                }
                CHECK(B.size()==now_b_size);
                CHECK(S.size()==now_s_size);
                // pre_score = now_score;
                inner_edge_num = temp_inner;
                P.insert(d);
                const auto& oes = graph_->GetOutgoingAdjList(d);
                if(oes.Size() < try_max_node){
                    for(auto& u : oes){
                        if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
                            D.push(u.neighbor);
                            visited_nodes.insert(u.neighbor);
                        }
                    }
                }
            }
        }

        // if(P.size() >= MIN_NODE_NUM){
        if(P.size() >= now_min_node_num){
            // debug
            // {
            //     if(B.size() * S.size() >= inner_edge_num*cost_ratio){
            //         LOG(INFO) << "now_s_size=" << S.size() << " now_b_size=" << B.size() << " temp_inner=" << inner_edge_num;
            //         LOG(INFO) << "last_s_size=" << last_s_size << " last_b_size=" << last_b_size << " last_ie_num=" << last_ie_num;
            //         // last_s_size=9 last_b_size=9 last_ie_num=16
            //         exit(0);
            //     }
            //     long long temp_ie_num = 0;
            //     vertex_t s;
            //     // inner edge, out bound edge
            //     for(auto v : P){
            //         const auto& oes = graph_->GetOutgoingAdjList(v);
            //         for(auto& e : oes){
            //             if(P.find(e.neighbor) != P.end()){
            //                 temp_ie_num++;
            //             }
            //         }
            //         s = v;
            //     }
            //     // inner bound node
            //     int b_num = 0;
            //     for(auto v : P){
            //         const auto& oes = graph_->GetOutgoingAdjList(v);
            //         for(auto& e : oes){
            //             if(P.find(e.neighbor) == P.end()){
            //                 b_num++;
            //                 break;
            //             }
            //         }
            //     }
            //     // source node
            //     int s_num = 0;
            //     for(auto v : P){
            //         const auto& oes = graph_->GetIncomingAdjList(v);
            //         for(auto& e : oes){
            //             if(P.find(e.neighbor) == P.end()){
            //                 s_num++;
            //                 break;
            //             }
            //         }
            //     }
            //     // CHECK_EQ(temp_ie_num, inner_edge_num);   //  (22 vs. 28)             
            //     CHECK_EQ(b_num, B.size());               
            //     CHECK_EQ(s_num, S.size());               
            //     if(temp_ie_num != inner_edge_num){
            //         std::cout << "------------\n" << std::endl;
            //         LOG(INFO) << "temp_ie_num=" << temp_ie_num << " inner_edge_num=" << inner_edge_num;
            //         std::cout << "--b:" << std::endl;
            //         for(auto b : B){
            //             std::cout << b.GetValue() << ", ";
            //         }
            //         std::cout << "\n--s: " << std::endl;
            //         for(auto s : S){
            //            std::cout << s.GetValue() << ", ";
            //         }
            //         std::cout << "\n";
            //         // inner edge, out bound edge
            //         long cnt = 0;
            //         for(auto v : P){
            //             const auto& oes = graph_->GetOutgoingAdjList(v);
            //             for(auto& e : oes){
            //                 if(P.find(e.neighbor) != P.end()){
            //                     cnt++;
            //                     std::cout << "inner_edge:" << cnt << ": " << v.GetValue() << " -> " << e.neighbor.GetValue() << std::endl;
            //                 }
            //             }
            //         }
            //         exit(0);
            //     }
            //     // vid_t src_id = Fc[s] < 0 ? (-Fc[s]-1) : Fc[s];
            //     // int s_num = supernode_source[src_id].size();
            //     // CHECK(s_num * b_num < temp_ie_num*cost_ratio);
            //     if(s_num * b_num >= temp_ie_num * cost_ratio){
            //         LOG(INFO) << "id=" << supernode_ids.size() << " s_num=" << s_num << " b_num=" << b_num << " temp_ie_num=" << temp_ie_num;
            //         // I0728 16:25:49.188849 51838 compressor_base.h:942] id=54 s_num=3 b_num=2 temp_ie_num=7
            //     }
            // }
            /* create a set of indexes for each entry */
            if(S.size() == 0){
                S.insert(source); // There is no entry point. At this time, the source point is used to form a super point.
            }
            int ids_id = -1;
            {
                std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                supernode_ids.emplace_back(P.begin(), P.end());
                ids_id = int(supernode_ids.size()) - 1;
                // supernode_bound_ids.emplace_back(O.begin(), O.end());
                supernode_source.emplace_back(S.begin(), S.end());
            }
            // CHECK(ids_id >= 0);
            for(auto u : P){
                Fc[u] = -(ids_id+1);
                id2spids[u] = ids_id;
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                // vid_t supernode_id = supernodes_num;
                // supernodes_num++;
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id;
            }
        }
    }

    /* 
       new cost mode + sort,
       The in-degree of source vertices is from small to large, 
       and the out-degree of candidate vertices is from small to large.
    */
    void parallel_find_multi_source_supernode_2_sort(const vertex_t source, vid_t begin, vid_t end, int now_min_node_num){
        std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        std::set<vertex_t> P; // inner node
        std::set<vertex_t> B; // belong to P, bound vertices
        // std::queue<vertex_t> D; // candidate queue
        std::priority_queue<std::pair<vertex_t, vid_t>, std::vector<std::pair<vertex_t, vid_t>>, cmp_pair_s2b> D; // candidate queue
        // std::unordered_set<vertex_t> visited_nodes;
        std::set<vertex_t> visited_nodes;
        fid_t fid_ = graph_->GetFragId(source);
        const int try_max_node = MAX_NODE_NUM * 1000; // 1 776 852
        const int try_max_step = MAX_NODE_NUM * 1000;
        const float cost_ratio = 1;

        // add source
        if(source.GetValue() >= begin && source.GetValue() < end && Fc[source] == FC_default_value && fid_ == graph_->GetFragId(source)){
            // D.push(source);
            D.push(std::pair<vertex_t, vid_t>(source, graph_->GetOutgoingAdjList(source).Size()));
            visited_nodes.insert(source);
        }

        vid_t inner_edge_num = 0;
        vid_t temp_inner = 0;
        float pre_score = 0; // pre compressed score
        int step = 0;
        long long last_s_size = 0;
        long long last_b_size = 0;
        long long last_ie_num = 0;
        while(!D.empty() && P.size() <= MAX_NODE_NUM){
            step++;
            // if(step > 1000){
            if(step > try_max_step){
                LOG(INFO) << "step>" << try_max_step << ", workid=" << comm_spec_.worker_id() << " D.size=" << D.size() << " P.size=" << 
                P.size();
                break;
            }
            vertex_t d = D.top().first;
            D.pop();
            // if d add to P:
            bool is_s = false;
            bool is_b = false;
            temp_inner = inner_edge_num;
            // update S/P/O
            std::unordered_set<vertex_t> temp_B; // need to delete
            std::unordered_set<vertex_t> temp_S; // need to delete
            vid_t now_s_size = S.size();
            vid_t now_b_size = B.size();
            for(auto& oe : graph_->GetOutgoingAdjList(d)){
                // update S
                if(S.find(oe.neighbor) != S.end()){
                    bool flag = true;
                    for(auto& ie : graph_->GetIncomingAdjList(oe.neighbor)){
                        if(P.find(ie.neighbor) == P.end() && ie.neighbor != d){
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        // temp_S.erase(v.neighbor);
                        temp_S.insert(oe.neighbor);
                        // now_s_size--;
                    }
                }
                // update temp_inner
                if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){ // self-edge
                    is_b = true;
                }
                else{
                    temp_inner++;
                }
            }
            for(auto& ie : graph_->GetIncomingAdjList(d)){
                // update B
                if(B.find(ie.neighbor) != B.end()){
                    bool flag = true;
                    for(auto& oe : graph_->GetOutgoingAdjList(ie.neighbor)){
                        if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){
                            flag = false;
                            break;
                        }
                    }
                    if(flag){
                        temp_B.insert(ie.neighbor);
                        // now_b_size--;
                    }
                }
                // update temp_inner
                // if(P.find(ie.neighbor) != P.end() || ie.neighbor == d){ // self-edge
                if(P.find(ie.neighbor) != P.end()){ // self-edge just only count once, it has been calculated in the out side.
                    temp_inner++;
                }
                else{
                    is_s = true;
                }
            }
            now_s_size -= temp_S.size();
            now_b_size -= temp_B.size();
            if(is_s){
                // temp_S.insert(d);
                now_s_size++;
            }
            if(is_b){
                // B.insert(d);
                now_b_size++;
            }

            // float now_score = temp_inner * 1.0 / (temp_S.size() * temp_O.size());
            // if((P.size() < 4) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            // if((P.size() < MIN_NODE_NUM-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            // if((P.size() < now_min_node_num-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
            if((P.size() < now_min_node_num-1) || (now_s_size * now_b_size < temp_inner * cost_ratio)){
            // if((now_score >= pre_score && P.size() < now_min_node_num-1) || now_score >= 1){
                /* update B */
                if(is_b){
                    B.insert(d);
                }
                for(auto v : temp_B){
                    B.erase(v);
                }
                /* update S */
                if(is_s){
                    S.insert(d);
                }
                for(auto v : temp_S){
                    S.erase(v);
                }
                CHECK(B.size()==now_b_size);
                CHECK(S.size()==now_s_size);
                // pre_score = now_score;
                inner_edge_num = temp_inner;
                P.insert(d);
                const auto& oes = graph_->GetOutgoingAdjList(d);
                if(oes.Size() < try_max_node){
                    for(auto& u : oes){
                        if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
                            // D.push(u.neighbor);
                            D.push(std::pair<vertex_t, vid_t>(u.neighbor, graph_->GetOutgoingAdjList(u.neighbor).Size()));
                            visited_nodes.insert(u.neighbor);
                        }
                    }
                }
            }
        }

        // if(P.size() >= MIN_NODE_NUM){
        if(P.size() >= now_min_node_num){
            /* create a set of indexes for each entry */
            if(S.size() == 0){
                S.insert(source); // There is no entry point. At this time, the source point is used to form a super point.
            }
            int ids_id = -1;
            {
                std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                supernode_ids.emplace_back(P.begin(), P.end());
                ids_id = int(supernode_ids.size()) - 1;
                // supernode_bound_ids.emplace_back(O.begin(), O.end());
                supernode_source.emplace_back(S.begin(), S.end());
            }
            // CHECK(ids_id >= 0);
            for(auto u : P){
                Fc[u] = -(ids_id+1);
                id2spids[u] = ids_id;
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                // vid_t supernode_id = supernodes_num;
                // supernodes_num++;
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id;
            }
        }
    }

    float check_score(vertex_t d, std::set<vertex_t>& S, std::set<vertex_t>& P, std::set<vertex_t>& B, float ring_weight, std::vector<short int>& is_s_vec, std::vector<short int>& is_b_vec, std::vector<std::unordered_set<vertex_t>>& temp_S_vec, std::vector<std::unordered_set<vertex_t>>& temp_B_vec, std::vector<vid_t>& temp_inner_vec, int i){
        // if d add to P:
        bool is_s = false;
        bool is_b = false;
        vid_t temp_inner = temp_inner_vec[i];
        // update S/P/O
        // std::unordered_set<vertex_t> temp_B; // need to delete
        // std::unordered_set<vertex_t> temp_S; // need to delete
        std::unordered_set<vertex_t>& temp_B = temp_B_vec[i];
        std::unordered_set<vertex_t>& temp_S = temp_S_vec[i];
        temp_B.clear();
        temp_S.clear();
        vid_t now_s_size = S.size();
        vid_t now_b_size = B.size();
        for(auto& oe : graph_->GetOutgoingAdjList(d)){
            // update S
            if(S.find(oe.neighbor) != S.end()){
                bool flag = true;
                for(auto& ie : graph_->GetIncomingAdjList(oe.neighbor)){
                    if(P.find(ie.neighbor) == P.end() && ie.neighbor != d){
                        flag = false;
                        break;
                    }
                }
                if(flag){
                    // temp_S.erase(v.neighbor);
                    temp_S.insert(oe.neighbor);
                    // now_s_size--;
                }
            }
            // update temp_inner
            if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){ // self-edge
                is_b = true;
            }
            else{
                // temp_inner++;
                temp_inner += ring_weight;
            }
        }
        for(auto& ie : graph_->GetIncomingAdjList(d)){
            // update B
            if(B.find(ie.neighbor) != B.end()){
                bool flag = true;
                for(auto& oe : graph_->GetOutgoingAdjList(ie.neighbor)){
                    if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){
                        flag = false;
                        break;
                    }
                }
                if(flag){
                    temp_B.insert(ie.neighbor);
                    // now_b_size--;
                }
            }
            // update temp_inner
            // if(P.find(ie.neighbor) != P.end() || ie.neighbor == d){ // self-edge
            if(P.find(ie.neighbor) != P.end()){ // self-edge just only count once, it has been calculated in the out side.
                temp_inner++;
            }
            else{
                is_s = true;
            }
        }
        now_s_size -= temp_S.size();
        now_b_size -= temp_B.size();
        if(is_s){
            // temp_S.insert(d);
            now_s_size++;
        }
        if(is_b){
            // B.insert(d);
            now_b_size++;
        }

        is_s_vec[i] = is_s;
        is_b_vec[i] = is_b;
        temp_inner_vec[i] = temp_inner;
        // LOG(INFO) << "i=" << i << " is_s=" << is_s << " is_b=" << is_b;  
        // return temp_inner * 1.0 / (now_s_size * now_b_size + 1e-3);
        // return (temp_inner * 1.0 - (now_s_size * now_b_size)) / temp_inner; // Reference: Graph Summarization with Bounded Error
        return temp_inner * 1.0 - (now_s_size * now_b_size);
    }

    /* new cost mode + sample*/
    void parallel_find_multi_source_supernode_2_sort_sample(const vertex_t source, vid_t begin, vid_t end, int now_min_node_num){
        std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        std::set<vertex_t> P; // inner node
        std::set<vertex_t> B; // belong to P, bound vertices
        // std::queue<vertex_t> D; // candidate queue
        // std::priority_queue<std::pair<vertex_t, vid_t>, std::vector<std::pair<vertex_t, vid_t>>, cmp_pair_s2b> D; // candidate queue
        // std::unordered_set<vertex_t> visited_nodes;
        Queue<vertex_t, vid_t> D;
        std::set<vertex_t> visited_nodes;
        fid_t fid_ = graph_->GetFragId(source);
        const int try_max_node = MAX_NODE_NUM * 1000; // 1 776 852
        const int try_max_step = MAX_NODE_NUM * 1000;
        const float obj_score = 1;
        int sample_size = 32;
        const int try_sample_cnt = 3;
        const float ring_weight = 1;

        /* step1: add source and 1-hop to P */
        if(source.GetValue() >= begin && source.GetValue() < end && Fc[source] == FC_default_value && fid_ == graph_->GetFragId(source)){
            P.insert(source);
            visited_nodes.insert(source);
            /* 1-hop */
            const auto& oes = graph_->GetOutgoingAdjList(source);
            if(oes.Size() < try_max_node){
                for(auto& u : oes){
                    if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
                        // P.insert(u.neighbor);
                        D.push(u.neighbor);
                        visited_nodes.insert(u.neighbor);
                        /* 2-hop */
                        // const auto& oes2 = graph_->GetOutgoingAdjList(u.neighbor);
                        // if(oes2.Size() < try_max_node){
                        //     for(auto& u : oes2){
                        //         if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
                        //             D.push(u.neighbor);
                        //             visited_nodes.insert(u.neighbor);
                        //         }
                        //     }
                        // }
                    }
                }
            }
        }

        vid_t inner_edge_num = 0;
        vid_t temp_inner = 0;

        /* update S, B, and inner_edge_num */
        for(auto u : P){
            bool f = true;
            for(auto& oe : graph_->GetOutgoingAdjList(u)){
                if(P.find(oe.neighbor) == P.end()){
                    if(f){
                        B.insert(u);
                        f = false;
                    }
                }
                else{
                    inner_edge_num++;
                }
            }
            for(auto& ie : graph_->GetIncomingAdjList(u)){
                if(P.find(ie.neighbor) == P.end()){
                    S.insert(u);
                    break;
                }
            }
        }

        float pre_score = 0; // pre compressed score
        int step = 0;
        long long last_s_size = 0;
        long long last_b_size = 0;
        long long last_ie_num = 0;
        /* step2: start sample from 2-hop */
        std::vector<float> scores(sample_size);
        std::vector<short int> is_s_vec(sample_size);
        std::vector<short int> is_b_vec(sample_size);
        std::vector<std::unordered_set<vertex_t>> temp_S_vec(sample_size);
        std::vector<std::unordered_set<vertex_t>> temp_B_vec(sample_size);
        std::vector<vid_t> temp_inner_vec(sample_size);
        std::vector<vid_t> sample;
        while(!D.empty() && P.size() <= MAX_NODE_NUM){
            step++;
            // if(step > 1000){
            if(step > try_max_step){
                LOG(INFO) << "step>" << try_max_step << ", workid=" << comm_spec_.worker_id() << " D.size=" << D.size() << " P.size=" << P.size();
                break;
            }
            // vertex_t d = D.top().first;
            // D.pop();

            vertex_t d;
            int max_id = -1;
            float max_score = 0;
            // if(P.size() < now_min_node_num - 1){
                bool finded = false;
                int cnt = 0;
                // LOG(INFO) << "start sample...";
                while(cnt < try_sample_cnt){
                    sample.clear();
                    D.sample(sample_size, sample);
                    int real_size = sample.size();
                    parallel_for(int i = 0; i < real_size; i++){
                    // for(int i = 0; i < real_size; i++){
                        // LOG(INFO) << "sample——Id：" << sample[i];
                        temp_inner_vec[i] = inner_edge_num;
                        // scores[i] = check_score(D.getById(sample[i]), S, P, B, inner_edge_num, ring_weight);
                        scores[i] = check_score(D.getById(sample[i]), S, P, B, ring_weight, is_s_vec, is_b_vec, temp_S_vec, temp_B_vec, temp_inner_vec, i);
                    }
                    max_score = -1e10;
                    for(int i = 0; i < real_size; i++){
                        // max_score = std::max(max_score, scores[i]);
                        if(max_score < scores[i]){
                            max_score = scores[i];
                            max_id = i;
                        }
                    }
                    // if(max_score >= obj_score || (P.size() < now_min_node_num-1)){ // 大于目标值
                    if((max_score >= 0 && max_score >= pre_score) || (P.size() < now_min_node_num-1)){ // 必须保持递增
                        pre_score = max_score;
                        d = D.getById(sample[max_id]);
                        // LOG(INFO) << "--find：" << d.GetValue();
                        D.erase(sample[max_id]);
                        finded = true;
                        break;
                    }
                    if(sample_size >= D.size()){
                        break;
                    }
                    cnt++;
                }
                // LOG(INFO) << "finish sample...";
                if(!finded){
                    // LOG(INFO) << "not find：";
                    break;
                }
                // LOG(INFO) << "find：" << d.GetValue();
            // }
            // else{
            //     d = D.front();
            //     D.pop();
            // }
            
            // if d add to P:
            // bool is_s = false;
            // bool is_b = false;
            // temp_inner = inner_edge_num;
            // // update S/P/O
            // std::unordered_set<vertex_t> temp_B; // need to delete
            // std::unordered_set<vertex_t> temp_S; // need to delete
            // vid_t now_s_size = S.size();
            // vid_t now_b_size = B.size();
            // for(auto& oe : graph_->GetOutgoingAdjList(d)){
            //     // update S
            //     if(S.find(oe.neighbor) != S.end()){
            //         bool flag = true;
            //         for(auto& ie : graph_->GetIncomingAdjList(oe.neighbor)){
            //             if(P.find(ie.neighbor) == P.end() && ie.neighbor != d){
            //                 flag = false;
            //                 break;
            //             }
            //         }
            //         if(flag){
            //             // temp_S.erase(v.neighbor);
            //             temp_S.insert(oe.neighbor);
            //             // now_s_size--;
            //         }
            //     }
            //     // update temp_inner
            //     if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){ // self-edge
            //         is_b = true;
            //     }
            //     else{
            //         // temp_inner++;
            //         temp_inner += ring_weight;
            //     }
            // }
            // for(auto& ie : graph_->GetIncomingAdjList(d)){
            //     // update B
            //     if(B.find(ie.neighbor) != B.end()){
            //         bool flag = true;
            //         for(auto& oe : graph_->GetOutgoingAdjList(ie.neighbor)){
            //             if(P.find(oe.neighbor) == P.end() && oe.neighbor != d){
            //                 flag = false;
            //                 break;
            //             }
            //         }
            //         if(flag){
            //             temp_B.insert(ie.neighbor);
            //             // now_b_size--;
            //         }
            //     }
            //     // update temp_inner
            //     // if(P.find(ie.neighbor) != P.end() || ie.neighbor == d){ // self-edge
            //     if(P.find(ie.neighbor) != P.end()){ // self-edge just only count once, it has been calculated in the out side.
            //         temp_inner++;
            //     }
            //     else{
            //         is_s = true;
            //     }
            // }
            // now_s_size -= temp_S.size();
            // now_b_size -= temp_B.size();
            // if(is_s){
            //     // temp_S.insert(d);
            //     now_s_size++;
            // }
            // if(is_b){
            //     // B.insert(d);
            //     now_b_size++;
            // }
            
            bool is_s = is_s_vec[max_id];
            bool is_b = is_b_vec[max_id];
            temp_inner = temp_inner_vec[max_id];
            std::unordered_set<vertex_t>& temp_B = temp_B_vec[max_id]; // need to delete
            std::unordered_set<vertex_t>& temp_S = temp_S_vec[max_id]; // need to delete

            // debug
            // {
            //     if(is_s != is_s_vec[max_id]){
            //         LOG(INFO) << "max_Id=" << max_id << std::endl;
            //         LOG(INFO) << "is_s=" << is_s << " is_s_vec=" << is_s_vec[max_id];
            //         exit(0);
            //     }
            //     if(is_b != is_b_vec[max_id]){
            //         LOG(INFO) << "is_b=" << is_b << " is_s_vec=" << is_b_vec[max_id];
            //         exit(0);
            //     }
            //     if(temp_B.size() != temp_B_vec[max_id].size()){
            //         for(auto b : B){
            //             std::cout << b.GetValue() << " ";
            //         }
            //         std::cout << "\ntb:\n";
            //         for(auto tb : temp_B_vec[max_id]){
            //             std::cout << tb.GetValue() << " ";
            //         }
            //         std::cout << "\n";
            //         exit(0);
            //     }
            //     if(temp_S.size() != temp_S_vec[max_id].size()){
            //         for(auto s : S){
            //             std::cout << s.GetValue() << " ";
            //         }
            //         std::cout << "\nts:\n";
            //         for(auto ts : temp_S_vec[max_id]){
            //             std::cout << ts.GetValue() << " ";
            //         }
            //         std::cout << "\n";
            //         exit(0);
            //     }
            // }

            // float now_score = temp_inner * 1.0 / (temp_S.size() * temp_O.size());
            // if((P.size() < now_min_node_num-1) || (max_score  >= obj_score)){
                /* update B */
                if(is_b){
                    B.insert(d);
                }
                for(auto v : temp_B){
                    B.erase(v);
                }
                /* update S */
                if(is_s){
                    S.insert(d);
                }
                for(auto v : temp_S){
                    S.erase(v);
                }
                // CHECK(B.size()==now_b_size);
                // CHECK(S.size()==now_s_size);
                // pre_score = now_score;
                inner_edge_num = temp_inner;
                P.insert(d);
                const auto& oes = graph_->GetOutgoingAdjList(d);
                if(oes.Size() < try_max_node){
                    for(auto& u : oes){
                        if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
                            D.push(u.neighbor);
                            // D.push(std::pair<vertex_t, vid_t>(u.neighbor, graph_->GetOutgoingAdjList(u.neighbor).Size()));
                            visited_nodes.insert(u.neighbor);
                        }
                    }
                }
            // }
            // LOG(INFO) << "finish a round...";
        }

        // if(P.size() >= MIN_NODE_NUM){
        if(P.size() >= now_min_node_num){
            // debug
            // {
            //     if(B.size() * S.size() >= inner_edge_num){
            //         LOG(INFO) << "now_s_size=" << S.size() << " now_b_size=" << B.size() << " temp_inner=" << inner_edge_num;
            //         LOG(INFO) << "last_s_size=" << last_s_size << " last_b_size=" << last_b_size << " last_ie_num=" << last_ie_num;
            //         // last_s_size=9 last_b_size=9 last_ie_num=16
            //         exit(0);
            //     }
            //     long long temp_ie_num = 0;
            //     vertex_t s;
            //     // inner edge, out bound edge
            //     for(auto v : P){
            //         const auto& oes = graph_->GetOutgoingAdjList(v);
            //         for(auto& e : oes){
            //             if(P.find(e.neighbor) != P.end()){
            //                 temp_ie_num++;
            //             }
            //         }
            //         s = v;
            //     }
            //     // inner bound node
            //     int b_num = 0;
            //     for(auto v : P){
            //         const auto& oes = graph_->GetOutgoingAdjList(v);
            //         for(auto& e : oes){
            //             if(P.find(e.neighbor) == P.end()){
            //                 b_num++;
            //                 break;
            //             }
            //         }
            //     }
            //     // source node
            //     int s_num = 0;
            //     for(auto v : P){
            //         const auto& oes = graph_->GetIncomingAdjList(v);
            //         for(auto& e : oes){
            //             if(P.find(e.neighbor) == P.end()){
            //                 s_num++;
            //                 break;
            //             }
            //         }
            //     }
            //     // CHECK_EQ(temp_ie_num, inner_edge_num);   //  (22 vs. 28)             
            //     // CHECK_EQ(b_num, B.size());               
            //     // CHECK_EQ(s_num, S.size());  //Check failed: s_num == S.size() (1 vs. 2)        
            //     if(temp_ie_num != inner_edge_num || s_num != S.size() || b_num != B.size()){
            //         std::cout << "------------\n" << std::endl;
            //         std::cout << "source:" << source.GetValue() << std::endl;
            //         LOG(INFO) << "s_num=" << s_num << " b_num=" << b_num;
            //         LOG(INFO) << "temp_ie_num=" << temp_ie_num << " inner_edge_num=" << inner_edge_num;
            //         std::cout << "\nB_num=" << B.size() << "\nb:\n";
            //         for(auto b : B){
            //             std::cout << b.GetValue() << ", ";
            //         }
            //         std::cout << "\nS_num=" << S.size() << "\ns:\n";
            //         for(auto s : S){
            //            std::cout << s.GetValue() << ", ";
            //         }
            //         std::cout << "\np_num=" << P.size() << "\np:\n";
            //         for(auto p : P){
            //             std::cout << p.GetValue() << " ";
            //         }
            //         std::cout << "\nreal_s:\n";
            //         for(auto v : P){
            //             const auto& oes = graph_->GetIncomingAdjList(v);
            //             for(auto& e : oes){
            //                 if(P.find(e.neighbor) == P.end()){
            //                     std::cout << v.GetValue() << ", ";
            //                     break;
            //                 }
            //             }
            //         }
            //         std::cout << "\nreal_b:\n";
            //         for(auto v : P){
            //             const auto& oes = graph_->GetOutgoingAdjList(v);
            //             for(auto& e : oes){
            //                 if(P.find(e.neighbor) == P.end()){
            //                     std::cout << v.GetValue() << ", ";
            //                     break;
            //                 }
            //             }
            //         }
            //         std::cout << "\n";
            //         // inner edge, out bound edge
            //         long cnt = 0;
            //         for(auto v : P){
            //             const auto& oes = graph_->GetOutgoingAdjList(v);
            //             for(auto& e : oes){
            //                 if(P.find(e.neighbor) != P.end()){
            //                     cnt++;
            //                     std::cout << "inner_edge:" << cnt << ": " << v.GetValue() << " -> " << e.neighbor.GetValue() << std::endl;
            //                 }
            //                 else{
            //                     std::cout << "out:" << cnt << ": " << v.GetValue() << " -> " << e.neighbor.GetValue() << std::endl;
            //                 }
            //             }
            //         }
            //         exit(0);
            //     }
            //     // vid_t src_id = Fc[s] < 0 ? (-Fc[s]-1) : Fc[s];
            //     // int s_num = supernode_source[src_id].size();
            //     // CHECK(s_num * b_num < temp_ie_num*cost_ratio);
            //     if(s_num * b_num >= temp_ie_num){
            //         LOG(INFO) << "id=" << supernode_ids.size() << " s_num=" << s_num << " b_num=" << b_num << " temp_ie_num=" << temp_ie_num;
            //         // I0728 16:25:49.188849 51838 compressor_base.h:942] id=54 s_num=3 b_num=2 temp_ie_num=7
            //     }
            // }
            
            /* create a set of indexes for each entry */
            if(S.size() == 0){
                S.insert(source); // There is no entry point. At this time, the source point is used to form a super point.
            }
            int ids_id = -1;
            {
                std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                supernode_ids.emplace_back(P.begin(), P.end());
                ids_id = int(supernode_ids.size()) - 1;
                // supernode_bound_ids.emplace_back(O.begin(), O.end());
                supernode_source.emplace_back(S.begin(), S.end());
            }
            // CHECK(ids_id >= 0);
            for(auto u : P){
                Fc[u] = -(ids_id+1);
                id2spids[u] = ids_id;
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                // vid_t supernode_id = supernodes_num;
                // supernodes_num++;
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id;
            }
        }
    }

    // void parallel_part_find_multi_source_supernode(const vertex_t source, vid_t partid){
    //     // std::unordered_set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
    //     // std::unordered_set<vertex_t> P; // inner node
    //     // std::unordered_set<vertex_t> O; // V^out
    //     std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
    //     std::set<vertex_t> P; // inner node
    //     std::set<vertex_t> O; // V^out
    //     std::queue<vertex_t> D; // candidate queue
    //     // std::unordered_set<vertex_t> visited_nodes;
    //     std::set<vertex_t> visited_nodes;
    //     fid_t fid_ = graph_->GetFragId(source);
    //     // add source
    //     visited_nodes.insert(source);
    //     P.insert(source);
    //     const auto& ies = graph_->GetIncomingAdjList(source);
    //     for(auto& e : ies){
    //         if(e.neighbor != source){
    //             S.insert(source);
    //             break;
    //         }
    //     }
    //     const auto& oes = graph_->GetOutgoingAdjList(source);
    //     for(auto& e : oes){
    //         auto& u = e.neighbor;
    //         if(graph_part[u.GetValue()] == partid && visited_nodes.find(u) == visited_nodes.end() && Fc[u] == FC_default_value && fid_ == graph_->GetFragId(u)){
    //             // D.push(u);
    //             // visited_nodes.insert(u);
    //             auto noe = graph_->GetOutgoingAdjList(u);
    //             if(noe.Size() <= MAX_NODE_NUM){ // filter powerlog node!!!
    //                 D.push(u);
    //                 visited_nodes.insert(u);
    //             }
    //         }
    //         if(O.find(u) == O.end() && P.find(u) == P.end()){
    //             O.insert(u);
    //         }
    //     }
    //     vid_t inner_edge_num = 0;
    //     vid_t bound_edge_num = O.size();
    //     vid_t temp_inner = 0;
    //     vid_t temp_bound = 0;
    //     float pre_score = 0; // pre compressed score
    //     int step = 0;
    //     while(!D.empty() && P.size() <= MAX_NODE_NUM){
    //         step++;
    //         // if(step > 1000){
    //         if(step > 5000){
    //             LOG(INFO) << "step>5000, workid=" << comm_spec_.worker_id() << " D.size=" << D.size() << " P.size=" << 
    //             P.size();
    //             // exit(0);
    //             break;
    //         }
    //         vertex_t d = D.front();
    //         D.pop();
    //         // if d add to P:
    //         bool is_s = false;
    //         temp_inner = inner_edge_num;
    //         temp_bound = bound_edge_num;
    //         for(auto& u : graph_->GetIncomingAdjList(d)){
    //             if(P.find(u.neighbor) != P.end()){
    //                 temp_bound--;
    //                 temp_inner++;
    //             }
    //             else{
    //                 is_s = true;
    //             }
    //         }
    //         for(auto& u : graph_->GetOutgoingAdjList(d)){
    //             if(P.find(u.neighbor) == P.end()){
    //                 temp_bound++;
    //             }
    //             else{
    //                 temp_inner++;
    //             }
    //         }
    //         // update S/P
    //         std::unordered_set<vertex_t> temp_O;
    //         std::unordered_set<vertex_t> temp_S;
    //         // temp_S.insert(S.begin(), S.end());
    //         vid_t now_s_size = S.size();
    //         vid_t now_o_size = O.size();
    //         for(auto& v : graph_->GetOutgoingAdjList(d)){
    //             if(S.find(v.neighbor) != S.end()){
    //                 bool flag = true;
    //                 for(auto& u : graph_->GetIncomingAdjList(v.neighbor)){
    //                     if(P.find(u.neighbor) == P.end() && u.neighbor != d){
    //                         flag = false;
    //                         break;
    //                     }
    //                 }
    //                 if(flag){
    //                     // temp_S.erase(v.neighbor);
    //                     temp_S.insert(v.neighbor);
    //                     now_s_size--;
    //                 }
    //             }
    //         }
    //         if(is_s){
    //             // temp_S.insert(d);
    //             now_s_size++;
    //         }
    //         // temp_O.insert(O.begin(), O.end());
    //         // temp_O.erase(d);
    //         now_o_size--; // delete d
    //         for(auto& u : graph_->GetOutgoingAdjList(d)){
    //             if(P.find(u.neighbor) == P.end() && O.find(u.neighbor) == O.end() && u.neighbor != d){
    //                 temp_O.insert(u.neighbor);
    //             }
    //         }
    //         now_o_size += temp_O.size();

    //         // float now_score = temp_inner * 1.0 / (temp_bound + P.size());
    //         // float now_score = temp_inner * 1.0 / (temp_S.size() * temp_O.size());
    //         // if((P.size() < 4) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
    //         // if((P.size() < MIN_NODE_NUM-1) || (temp_S.size() * temp_O.size() < temp_inner * 1)){
    //         if((P.size() < MIN_NODE_NUM-1) || (now_s_size * now_o_size < temp_inner * 1)){
    //         // if((now_score >= pre_score && P.size() < MIN_NODE_NUM-1) || now_score >= 1){
    //             /* update O */
    //             // temp_O.swap(O);
    //             O.erase(d);
    //             O.insert(temp_O.begin(), temp_O.end());
    //             /* update S */
    //             // temp_S.swap(S);
    //             if(is_s){
    //                 S.insert(d);
    //             }
    //             for(auto v : temp_S){
    //                 S.erase(v);
    //             }
    //             // pre_score = now_score;
    //             inner_edge_num = temp_inner;
    //             bound_edge_num = temp_bound;
    //             P.insert(d);
    //             for(auto& u : graph_->GetOutgoingAdjList(d)){
    //                 // if(u.neighbor.GetValue() >= begin && u.neighbor.GetValue() < end && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
    //                 if(graph_part[u.neighbor.GetValue()] == partid && Fc[u.neighbor] == FC_default_value && P.find(u.neighbor) == P.end() && visited_nodes.find(u.neighbor) == visited_nodes.end() && fid_ == graph_->GetFragId(u.neighbor)){
    //                     // D.push(u.neighbor);
    //                     // visited_nodes.insert(u.neighbor);
    //                     auto noe = graph_->GetOutgoingAdjList(u.neighbor);
    //                     if(noe.Size() <= MAX_NODE_NUM && D.size() <= MAX_NODE_NUM * 10){ // filter some nodes that hava big degree!!!
    //                         D.push(u.neighbor);
    //                         visited_nodes.insert(u.neighbor);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // if(P.size() >= MIN_NODE_NUM){
    //     if(P.size() >= MIN_NODE_NUM){
    //         // if(bound_edge_num > 500){
    //         //     LOG(INFO) << "--pre_score=" << pre_score;
    //         // }
    //         /* create a set of indexes for each entry */
    //         if(S.size() == 0){
    //             S.insert(source); // There is no entry point. At this time, the source point is used to form a super point.
    //         }
    //         int ids_id = -1;
    //         {
    //             std::unique_lock<std::mutex> lk(supernode_ids_mux_);
    //             supernode_ids.emplace_back(P.begin(), P.end());
    //             ids_id = int(supernode_ids.size()) - 1;
    //             // supernode_bound_ids.emplace_back(O.begin(), O.end());
    //             supernode_source.emplace_back(S.begin(), S.end());
    //         }
    //         // CHECK(ids_id >= 0);
    //         for(auto u : P){
    //             Fc[u] = -(ids_id+1);
    //             id2spids[u] = ids_id;
    //         }
    //         for(auto src : S){
    //             Fc[src] = ids_id;
    //             /* build a supernode */
    //             // vid_t supernode_id = supernodes_num;
    //             // supernodes_num++;
    //             vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
    //             Fc_map[src] = supernode_id;
    //             supernodes[supernode_id].id = src;
    //             supernodes[supernode_id].ids = ids_id;
    //         }
    //     }
    // }


    void inc_compress(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges){
        fid_t fid = graph_->fid();
        auto vm_ptr = graph_->vm_ptr();
        inccalculate_spnode_ids.clear();
        recalculate_spnode_ids.clear();
        vid_t add_num = 0;
        vid_t del_num = 0;
        LOG(INFO) << "spnode_num=" << supernodes_num;
        LOG(INFO) << "deal deleted_edges...";

        for(auto& pair : deleted_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(graph_->Gid2Vertex(u_gid, u));
            if(u_fid == fid && Fc[u] != FC_default_value){
                // LOG(INFO) << graph_->GetId(u);
                vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                for(auto source : supernode_source[src_id]){
                    inccalculate_spnode_ids.insert(source.GetValue());
                }
            }
            vertex_t v;
            CHECK(graph_->Gid2Vertex(v_gid, v));
            if(v_fid == fid && Fc[v] != FC_default_value){
                // vid_t del_id = Fc_map[Fc[v][0]];
                vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                std::vector<vertex_t>& src = supernode_source[src_id];
                vid_t del_id = Fc_map[src[0]];
                supernode_t& spnode = supernodes[del_id];
                const vid_t ids_id = this->id2spids[spnode.id];
                if(ids_id != this->id2spids[u] && src.size() > 1){
                    CHECK(Fc[v] >= 0);
                    const auto& ies = graph_->GetIncomingAdjList(v);
                    bool hava_out_inadj = false;
                    // for(auto& e : ies){
                    //     auto& nb = e.neighbor;
                    //     if(nb != u && ids_id != this->id2spids[nb]){
                    //         hava_out_inadj = true;
                    //         break;
                    //     }
                    // }
                    /*-----parallel-----*/
                        auto out_degree = ies.Size();
                        auto it = ies.begin();
                        granular_for(j, 0, out_degree, (out_degree > 1024), {
                            auto& e = *(it + j);
                            auto& nb = e.neighbor;
                            if(nb != u && ids_id != this->id2spids[nb]){
                                hava_out_inadj = true;
                                // break;
                            }
                        })
                    if(hava_out_inadj == false){
                        delete_supernode(v);
                        del_num++;
                    }
                }
            }
        }
        LOG(INFO) << "deal added_edges...";
        for(auto& pair : added_edges){
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(graph_->Gid2Vertex(u_gid, u));
            if(u_fid == fid && Fc[u] != FC_default_value){
                // LOG(INFO) << graph_->GetId(u);
                // for(auto source: Fc[u]){
                vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                for(auto source : supernode_source[src_id]){
                    inccalculate_spnode_ids.insert(source.GetValue());
                }
            }
            vertex_t v;
            CHECK(graph_->Gid2Vertex(v_gid, v));
            if(v_fid == fid && Fc[v] != FC_default_value){
                // supernode_t& spnode = supernodes[Fc_map[Fc[v][0]]];
                vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                std::vector<vertex_t>& src = supernode_source[src_id];
                supernode_t& spnode = supernodes[Fc_map[src[0]]];
                auto& spids = supernode_ids[spnode.ids];
                const vid_t ids_id = this->id2spids[spnode.id];
                
                if(Fc[v] < 0 && ids_id != this->id2spids[u]){ // not a source, build a new spnode
                    // for(auto u : spids){
                    //     Fc[u] = src_id;
                    // }
                    // std::swap(Fc[v][0], Fc[v][int(Fc[v].size())-1]);
                    Fc[v] = src_id;
                    this->supernode_source[src_id].emplace_back(v);
                    // build a new spnode idnex
                    vid_t supernode_id = supernodes_num;
                    Fc_map[v] = supernode_id;
                    supernodes[supernode_id].id = v;
                    // supernodes[supernode_id].ids.insert(supernodes[supernode_id].ids.begin(), spnode.ids.begin(), spnode.ids.end());
                    supernodes[supernode_id].ids = spnode.ids;
                    supernodes_num++;

                    recalculate_spnode_ids.insert(v.GetValue());
                    add_num++;
                }
            }
        }
        // for(auto u : inccalculate_spnode_ids){
        //     LOG(INFO) << "inccalculate_spnode_ids:" << u << ":" << graph_->GetId(vertex_t(u));
        // }
        LOG(INFO) << "spnode_num=" << supernodes_num << " inccalculate_spnode_ids.size=" << inccalculate_spnode_ids.size() << " recalculate_spnode_ids.size=" << recalculate_spnode_ids.size() << " %=" << ((inccalculate_spnode_ids.size()+recalculate_spnode_ids.size())*1.0/supernodes_num);
        LOG(INFO) << "add_num=" << add_num << " del_num=" << del_num;
    }

    void parallel_inc_compress(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges){
        fid_t fid = graph_->fid();
        auto vm_ptr = graph_->vm_ptr();
        inccalculate_spnode_ids.clear();
        recalculate_spnode_ids.clear();
        vid_t add_num = 0;
        vid_t del_num = 0;
        LOG(INFO) << "spnode_num=" << supernodes_num;
        LOG(INFO) << "deal deleted_edges...";
        {
            /* parallel */
            int thread_num = FLAGS_app_concurrency > 0 ? FLAGS_app_concurrency : 4;
            std::vector<std::unordered_set<vid_t>> inc_temp;
            std::vector<std::unordered_set<vid_t>> del_temp;
            inc_temp.resize(thread_num);
            del_temp.resize(thread_num);
            this->ForEachIndex(deleted_edges.size(), [this, &vm_ptr, &inc_temp, &del_temp, &deleted_edges](int tid, vid_t begin, vid_t end) {
                fid_t fid = graph_->fid();
                // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                double start = GetCurrentTime();
                for(vid_t i = begin; i < end; i++){
                    auto& pair = deleted_edges[i];
                    auto u_gid = pair.first;
                    auto v_gid = pair.second;
                    fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                        v_fid = vm_ptr->GetFidFromGid(v_gid);
                    // u -> v
                    // LOG(INFO) << u_gid << "->" << v_gid;
                    vertex_t u;
                    CHECK(graph_->Gid2Vertex(u_gid, u));
                    if(u_fid == fid && Fc[u] != FC_default_value){
                        vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                        for(auto source : supernode_source[src_id]){
                            inc_temp[tid].insert(source.GetValue());
                        }
                    }
                    vertex_t v;
                    CHECK(graph_->Gid2Vertex(v_gid, v));
                    if(v_fid == fid && Fc[v] != FC_default_value){
                        vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                        std::vector<vertex_t>& src = supernode_source[src_id];
                        vid_t del_id = Fc_map[src[0]];
                        supernode_t& spnode = supernodes[del_id];
                        const vid_t ids_id = this->id2spids[spnode.id];
                        if(Fc[v] >= 0 && ids_id != this->id2spids[u] && src.size() > 1){
                            CHECK(Fc[v] >= 0);
                            // CHECK(Fc[v][0] == v);
                            const auto& ies = graph_->GetIncomingAdjList(v);
                            bool hava_out_inadj = false;
                            /*-----parallel-----*/
                                auto out_degree = ies.Size();
                                auto it = ies.begin();
                                granular_for(j, 0, out_degree, (out_degree > 1024), {
                                    auto& e = *(it + j);
                                    auto& nb = e.neighbor;
                                    if(nb != u && ids_id != this->id2spids[nb]){
                                        hava_out_inadj = true;
                                        // break;
                                    }
                                })
                            if(hava_out_inadj == false){
                                del_temp[tid].insert(v.GetValue());
                            }
                        }
                    }
                }
                // LOG(INFO) << "tid=" << tid << " finish! time=" << (GetCurrentTime()-start);
                }, thread_num
            );
            double start = GetCurrentTime();
            std::vector<vid_t> del_ids;
            for(int i = 0; i < thread_num; i++){ // 尝试改为bitset!!!
                // double start = GetCurrentTime();
                inccalculate_spnode_ids.insert(inc_temp[i].begin(), inc_temp[i].end());
                del_ids.insert(del_ids.end(), del_temp[i].begin(), del_temp[i].end());
                // LOG(INFO) << "---time=" << (GetCurrentTime()-start);
            }
            LOG(INFO) << "merge id---time=" << (GetCurrentTime()-start);
            vid_t count_id = 0;
            vid_t del_ids_num = del_ids.size();
            start = GetCurrentTime();
            this->ForEach(del_ids_num, [this, &count_id, &del_ids, &del_ids_num](int tid) {
                // LOG(INFO) << "build index, tid=" << tid << " begin..." << del_ids_num;
                int i = 0, cnt = 0, step = 1;
                while(i < del_ids_num){
                    i = __sync_fetch_and_add(&count_id, 1);
                    if(i < del_ids_num){
                        vertex_t u(del_ids[i]);
                        delete_supernode(u);
                    }
                }
                // LOG(INFO) << "tid=" << tid << " finish! cnt=" << cnt;
                }, thread_num
            );
            LOG(INFO) << "del supnode---time=" << (GetCurrentTime()-start);
        }

        // for(auto& pair : deleted_edges) {
        //     auto u_gid = pair.first;
        //     auto v_gid = pair.second;
        //     fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
        //           v_fid = vm_ptr->GetFidFromGid(v_gid);
        //     // u -> v
        //     // LOG(INFO) << u_gid << "->" << v_gid;
        //     vertex_t u;
        //     CHECK(graph_->Gid2Vertex(u_gid, u));
        //     if(u_fid == fid && Fc[u.GetValue()].size() > 0){
        //         // LOG(INFO) << graph_->GetId(u);
        //         for(auto source: Fc[u.GetValue()]){
        //             inccalculate_spnode_ids.insert(source.GetValue());
        //         }
        //     }
        //     vertex_t v;
        //     CHECK(graph_->Gid2Vertex(v_gid, v));
        //     if(v_fid == fid && Fc[v.GetValue()].size() > 0){
        //         vid_t del_id = Fc_map[Fc[v.GetValue()][0]];
        //         supernode_t& spnode = supernodes[del_id];
        //         // auto& spids = supernode_ids[spnode.ids];
        //         const vid_t ids_id = this->id2spids[spnode.id];
        //         // if(std::find(spids.begin(), spids.end(), u) == spids.end() && Fc[v.GetValue()].size() > 1){
        //         if(ids_id != this->id2spids[u] && Fc[v.GetValue()].size() > 1){
        //             CHECK(Fc[v.GetValue()][0] == v);
        //             const auto& ies = graph_->GetIncomingAdjList(v);
        //             bool hava_out_inadj = false;
        //             // for(auto& e : ies){
        //             //     auto& nb = e.neighbor;
        //             //     if(nb != u && ids_id != this->id2spids[nb]){
        //             //         hava_out_inadj = true;
        //             //         break;
        //             //     }
        //             // }
        //             /*-----parallel-----*/
        //                 auto out_degree = ies.Size();
        //                 auto it = ies.begin();
        //                 granular_for(j, 0, out_degree, (out_degree > 1024), {
        //                     auto& e = *(it + j);
        //                     auto& nb = e.neighbor;
        //                     if(nb != u && ids_id != this->id2spids[nb]){
        //                         hava_out_inadj = true;
        //                         // break;
        //                     }
        //                 })
        //             if(hava_out_inadj == false){
        //                 delete_supernode(v);
        //             }
        //         }
        //     }
        // }
        LOG(INFO) << "deal added_edges...";
        {
            double start = GetCurrentTime();
            for(auto& pair : added_edges){
                auto u_gid = pair.first;
                auto v_gid = pair.second;
                fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                    v_fid = vm_ptr->GetFidFromGid(v_gid);
                // u -> v
                // LOG(INFO) << u_gid << "->" << v_gid;
                vertex_t u;
                CHECK(graph_->Gid2Vertex(u_gid, u));
                if(u_fid == fid && Fc[u] != FC_default_value){
                    vid_t src_id = Fc[u] < 0 ? (-Fc[u]-1) : Fc[u];
                    // CHECK(src_id < supernode_source.size());
                    for(auto source : supernode_source[src_id]){
                        inccalculate_spnode_ids.insert(source.GetValue());
                    }
                }
                vertex_t v;
                CHECK(graph_->Gid2Vertex(v_gid, v));
                if(v_fid == fid && Fc[v] != FC_default_value){
                    vid_t src_id = Fc[v] < 0 ? (-Fc[v]-1) : Fc[v];
                    // CHECK(src_id < supernode_source.size());
                    std::vector<vertex_t>& src = supernode_source[src_id];
                    // CHECK(src.size() > 0);
                    // CHECK(Fc_map[src[0]] < supernodes_num);
                    supernode_t& spnode = supernodes[Fc_map[src[0]]];
                    auto& spids = supernode_ids[spnode.ids];
                    const vid_t ids_id = this->id2spids[spnode.id];
                    if(Fc[v] < 0 && ids_id != this->id2spids[u]){ // not a source, build a new spnode
                        Fc[v] = src_id;
                        this->supernode_source[src_id].emplace_back(v);
                        // build a new spnode idnex
                        vid_t supernode_id = supernodes_num;
                        Fc_map[v] = supernode_id;
                        supernodes[supernode_id].id = v;
                        // supernodes[supernode_id].ids.insert(supernodes[supernode_id].ids.begin(), spnode.ids.begin(), spnode.ids.end());
                        supernodes[supernode_id].ids = spnode.ids;
                        supernodes_num++;

                        recalculate_spnode_ids.insert(v.GetValue());
                    }
                }
            }
            LOG(INFO) << "add edge---time=" << (GetCurrentTime()-start);
        }
        // for(auto u : inccalculate_spnode_ids){
        //     LOG(INFO) << "inccalculate_spnode_ids:" << u << ":" << graph_->GetId(vertex_t(u));
        // }
        LOG(INFO) << "spnode_num=" << supernodes_num << " inccalculate_spnode_ids.size=" << inccalculate_spnode_ids.size() << " recalculate_spnode_ids.size=" << recalculate_spnode_ids.size();
        LOG(INFO) << "#spn_update_rate: " << ((inccalculate_spnode_ids.size()+recalculate_spnode_ids.size())*1.0/supernodes_num);
    }

    template<typename T = vertex_t>
    bool remove_array(std::vector<T> &array, T v){
        typename std::vector<T>::iterator it = std::find(array.begin(), array.end(), v);
        if(it == array.end()){
            return false;
        }
        std::swap(*it, array.back());
        array.pop_back();
        return true;
    }

    /**
     * Delete the super node, and note that the space of the point in the array 
     * is not released.
     * 从supernode_source会删除该source点
    */
    void delete_supernode(const vertex_t source){
        vid_t del_id = Fc_map[source]; // Get the index of spnode in the array
        supernode_t &spnode_v = supernodes[del_id];
        // updata supernode_source
        // for(auto u : supernode_ids[spnode_v.ids]){
        //     CHECK(remove_array(Fc[u.GetValue()], source));
        //     // id2spids[u] = ID_default_value;  // The default value is assigned only when there is no set of indexes
        // }
        // delete spnode
        std::unique_lock<std::mutex> lk(supernode_ids_mux_);
        vid_t src_id = Fc[source];
        if(src_id < 0 || supernode_source[src_id].size() <= 1){
            return;
        }
        Fc_map[spnode_v.id] = ID_default_value;
        spnode_v.clear();
        CHECK(src_id >= 0);
        Fc[source] = -(src_id+1);
        CHECK(remove_array(supernode_source[src_id], source));
        supernode_t &spnode_end = supernodes[supernodes_num-1];
        // updata Fc_map
        Fc_map[spnode_end.id] = del_id;
        // clear supernode 
        spnode_v.swap(spnode_end);
        supernodes_num--;
    }

    /**
     * 删除一个cluster, 需要回收哪些东西呢？
     * 1. 首先回收所有以入口点建立的supernode;
     * 2. 针对原始的cluster进行回收:
     * 3. 需要考虑cluster删除后，内部点v将变成外部点，而这些点可能是其它cluster内
     *    Mirrro点的Master, 所以，如果是Master，则需要将其放入其中一个Mirror所在
     *    的cluster中，并将其中的mirror_id替换掉.
    */
    void delete_cluster(const vid_t cid) {
      // LOG(INFO) << "-============================ cid=" << cid;
      // clear supnode of this cluster
      for (auto source : supernode_source[cid]) {
        // LOG(INFO) << " delete source oid=" << v2Oid(source);
        vid_t del_spid = Fc_map[source]; // Get the index of spnode in the array
        delete_supernode(del_spid);
        // LOG(INFO) << "    del_spid=" << del_spid;
      }
      for (auto source : cluster_in_mirror_ids[cid]) {
        // LOG(INFO) << " delete source mid=" << v2Oid(source);
        vid_t del_spid = Fc_map[source]; // Get the index of spnode in the array
        delete_supernode(del_spid);
        // LOG(INFO) << "    del_spid=" << del_spid;
      }
      // LOG(INFO) << "---------";
      // clear all info of cluster
      auto& all_nodes = cluster_ids[cid]; // include mirror
      for (auto v : all_nodes) {
        if (v.GetValue() < this->old_node_num) { // real node
          // master vertex
          Fc[v] = FC_default_value;
          Fc_map[v] = ID_default_value;
          id2spids[v] = ID_default_value;
        } else {
          // mirror
          Fc_map[v] = ID_default_value;
          id2spids[v] = ID_default_value;
        }
        supernode_out_bound[v.GetValue()] = false;
      }
      // clear mirror info in master cluster
      for (auto mid_v : cluster_in_mirror_ids[cid]) {
        // 将mid所在cluster的mid全部改成v
        auto master_vid = mirrorid2vid[mid_v];
        CHECK(remove_array<vid_t>(vid2in_mirror_mids[master_vid.GetValue()], 
                                    mid_v.GetValue()));
        CHECK(remove_array<vid_t>(
                    vid2in_mirror_cluster_ids[master_vid.GetValue()], cid));
      }
      for (auto mid_v : cluster_out_mirror_ids[cid]) {
        // 将mid所在cluster的mid全部改成v
        auto master_vid = mirrorid2vid[mid_v];
        CHECK(remove_array<vid_t>(vid2out_mirror_mids[master_vid.GetValue()], 
                                    mid_v.GetValue()));
      }

      // LOG(INFO) << "---------";
      auto& nodes_no_mirror = supernode_ids[cid];
      // supernode_ids[cid].clear(); 用完才能删除
      cluster_ids[cid].clear();
      supernode_source[cid].clear();
      cluster_out_mirror_ids[cid].clear();
      cluster_in_mirror_ids[cid].clear();
      supernode_out_mirror[cid].clear();
      supernode_in_mirror[cid].clear();
      // LOG(INFO) << "---------";
      // 需要将master的权限交给其它mirror。
      // 对于需要删除的master点，我们应该将其放入某个Mirror所在的cluster中。
      // 通过Mirror找master
      for (auto v : nodes_no_mirror) {
        bool used = false;
        // LOG(INFO) << " ======= v=" << v2Oid(v);
        for (auto t : vid2in_mirror_mids[v.GetValue()]) {
          auto mid = vertex_t(t);
          // 将mid所在cluster的mid全部改成v
          vid_t cid = id2spids[mid];
          // LOG(INFO) << " delete mid=" << t;
          // LOG(INFO) << " delete cid=" << cid;
          // LOG(INFO) << " delete v=" << v2Oid(v);
          Fc[v] = -(cid+1); // 先标记为内部点
          id2spids[v] = cid;
          used = true;
          for (auto &u : supernode_source[cid]) {
            if (u == mid) {
              u = v;
              Fc_map[v] = Fc_map[u];
              Fc[v] = cid;
              break;
            }
          }
          for (auto &u : cluster_ids[cid]) {
            if (u == mid) {
              u = v;
              break;
            }
          }
          // 加入cluster
          supernode_ids[cid].emplace_back(v);
          cluster_ids[cid].emplace_back(v);
          // 剔除mirror点
          // for (auto mv : cluster_in_mirror_ids[cid]) {
          //   LOG(INFO) << "cluster_in_mirror_ids[cid]: " << mv.GetValue();
          // }
          CHECK(remove_array(cluster_in_mirror_ids[cid], mid));
          supernode_in_mirror[cid].erase(v);
          CHECK(remove_array<vid_t>(vid2in_mirror_mids[v.GetValue()], 
                                    mid.GetValue()));
          CHECK(remove_array<vid_t>(vid2in_mirror_cluster_ids[v.GetValue()], 
                                    cid));
          // 检查out-mirror中是否有V点的Mirror
          for (auto out_mirror : cluster_out_mirror_ids[cid]) {
            if (mirrorid2vid[out_mirror].GetValue() == v.GetValue()) {
              CHECK(remove_array(cluster_out_mirror_ids[cid], out_mirror));
              CHECK(supernode_out_mirror[cid].erase(v));
              CHECK(remove_array<vid_t>(vid2out_mirror_mids[v.GetValue()], 
                                        out_mirror.GetValue()));
              break;
            }
          }
          break; // v仅仅能够将一个Mirror转为Master
        }
        // 如果已经放进了in-mirror所在的cluster中，则无须考虑放入out-mirror中
        if (used == true) {
          continue;
        }
        for (auto t : vid2out_mirror_mids[v.GetValue()]) {
          auto mid = vertex_t(t);
          // 将mid所在cluster的mid全部改成v
          vid_t cid = id2spids[mid];
          // LOG(INFO) << " delete mid=" << t;
          // LOG(INFO) << " delete cid=" << cid;
          // LOG(INFO) << " delete v=" << v2Oid(v);
          Fc[v] = -(cid+1); // 先标记为内部点
          id2spids[v] = cid;
          used = true;
          for (auto &u : supernode_source[cid]) {
            if (u == mid) {
              u = v;
              Fc_map[v] = Fc_map[u];
              Fc[v] = cid;
              break;
            }
          }
          for (auto &u : cluster_ids[cid]) {
            if (u == mid) {
              u = v;
              break;
            }
          }
          // 加入cluster
          supernode_ids[cid].emplace_back(v);
          cluster_ids[cid].emplace_back(v);
          // 剔除mirror点
          // for (auto mv : cluster_out_mirror_ids[cid]) {
          //   LOG(INFO) << "=cluster_out_mirror_ids[cid]: " << mv.GetValue();
          // }
          CHECK(remove_array(cluster_out_mirror_ids[cid], mid));
          supernode_out_mirror[cid].erase(v);
          CHECK(remove_array<vid_t>(vid2out_mirror_mids[v.GetValue()], 
                                    mid.GetValue()));
          // 检查in-mirror中是否有V点的Mirror
          for (auto in_mirror : cluster_in_mirror_ids[cid]) {
            if (mirrorid2vid[in_mirror].GetValue() == v.GetValue()) {
              CHECK(remove_array(cluster_in_mirror_ids[cid], in_mirror));
              CHECK(supernode_in_mirror[cid].erase(v));
              CHECK(remove_array<vid_t>(vid2in_mirror_mids[v.GetValue()], 
                                        in_mirror.GetValue()));
              CHECK(remove_array<vid_t>(vid2in_mirror_cluster_ids[v.GetValue()], 
                                        cid));
              break;
            }
          }
          break; // v仅仅能够将一个Mirror转为Master
        }
      }
      supernode_ids[cid].clear();
    }

    /**
     * Delete the super node, and note that the space of the point in the array is not released
     * 没有supernode_source中删除这个入口点
    */
    void delete_supernode(const vid_t del_spid){
        // LOG(INFO) << "  del_spid=" << del_spid << " sp_num=" << supernodes_num;
        supernode_t &spnode_v = supernodes[del_spid];
        vertex_t source = spnode_v.id;
        vid_t spids_id = spnode_v.ids;
        // LOG(INFO) << "  source=" << v2Oid(source);
        // delete spnode
        std::unique_lock<std::mutex> lk(supernode_ids_mux_);
        if(spids_id < 0){
            LOG(INFO) << "error.";
            exit(0);
        }
        CHECK(spids_id >= 0);
        // LOG(INFO) << " -----";
        supernode_t &spnode_end = supernodes[supernodes_num-1];
        // LOG(INFO) << " -----spnode_end.id=" << spnode_end.id.GetValue();
        // updata Fc_map
        Fc_map[spnode_end.id] = del_spid;
        // updata shortcuts 
        // LOG(INFO) << " -----spnode_end.ids=" << spnode_end.ids;
        vid_t vid = spnode_end.id.GetValue();
        if (vid >= this->old_node_num) { // mirror to master
          vid = this->mirrorid2vid[spnode_end.id].GetValue();
        }
        // LOG(INFO) << " exchage spid=" << this->shortcuts[vid][spnode_end.ids]
        //           << ", " << del_spid;
        this->shortcuts[vid][spnode_end.ids] = del_spid;
        // clear supernode 
        spnode_v.swap(spnode_end);
        supernodes_num--;
    }

    /**
     * use metis to cut graph
    */
    // void metis_cut(idx_t nParts, float tolerance_rate){
    //     auto inner_vertices = graph_->InnerVertices();
    //     LOG(INFO) << "start cut graph...";
    //     std::vector<idx_t> xadj(0);
    //     std::vector<idx_t> adjncy(0); // csr of graph
    //     std::vector<float> ubvec(nParts, tolerance_rate); // the allowed load imbalance tolerance for each constraint.
    //     // std::vector<idx_t> adjwgt(0); // weight
    //     fid_t fid_ = comm_spec_.fid();
    //     idx_t a, w;
    //     // vid_t all_num = graph_->GetVerticesNum();
    //     // int i = 0;
    //     for(auto u : inner_vertices){
    //         xadj.push_back(adjncy.size()); //csr: row offsets
    //         for(auto& e : graph_->GetOutgoingAdjList(u)){
    //             if(graph_->GetFragId(e.neighbor) == fid_){
    //                 adjncy.push_back(e.neighbor.GetValue()); // column indices
    //             }
    //         }
    //         // change to undirected graph
    //         for(auto& e : graph_->GetIncomingAdjList(u)){
    //             if(graph_->GetFragId(e.neighbor) == fid_){
    //                 adjncy.push_back(e.neighbor.GetValue()); // column indices
    //             }
    //         }
    //     }
    //     xadj.push_back(adjncy.size());
    //     func(xadj, adjncy, METIS_PartGraphKway, nParts, ubvec);
    // }

    // void func(std::vector<idx_t> &xadj, std::vector<idx_t> &adjncy, decltype(METIS_PartGraphKway) *METIS_PartGraphFunc, idx_t nParts_p, std::vector<float> ubvec) {
    //     idx_t nVertices = xadj.size() - 1; // node number
    //     idx_t nWeights = 1;                // node weight
    //     idx_t nParts = nParts_p;           // part number
    //     idx_t objval;                      // object function value, this variable stores the edge-cut or the total communication volume ofthe partitioning solution. 
    //     graph_part.resize(nVertices, 0);   // result
    //     LOG(INFO) << "---cuting....nParts=" << nParts << " nVertices=" << nVertices;
    //     int ret = METIS_PartGraphFunc(&nVertices, &nWeights, xadj.data(), adjncy.data(),
    //         NULL, NULL, NULL, &nParts, NULL,
    //         ubvec.data(), NULL, &objval, graph_part.data());

    //     if (ret != rstatus_et::METIS_OK){ 
    //         LOG(INFO) << "METIS_ERROR"; 
    //     }
    //     LOG(INFO) << "METIS_OK";
    //     LOG(INFO) << "objval: " << objval;

    //     /*
    //     // build supernodes: Make each partition a super point
    //     supernode_ids.resize(nParts);
    //     supernode_source.resize(nParts);
    //     // supernode_bound_ids.resize(nParts);
    //     int thread_num = NUM_THREADS;
    //     // int thread_num = 2; // 测试
    //     // Simulate thread pool
    //     vid_t spnode_id = 0;
    //     std::vector<vertex_t> S[thread_num];// belong to P, with vertices of incoming edges from the outside
    //     std::vector<vertex_t> P[thread_num];// inner node
    //     ForEach(nParts, [this, &spnode_id, &nParts, &P, &S](int tid) {
    //         LOG(INFO) << "build spnode, tid=" << tid << " begin..." << this->supernodes_num;
    //         int i = 0, cnt = 0, step = 1;
    //         vid_t MAXLEN = graph_->GetVerticesNum() / (omp_get_max_threads() * 10);
    //         P[tid].reserve(MAXLEN);
    //         S[tid].reserve(MAXLEN);
    //         while(i < nParts){
    //             i = __sync_fetch_and_add(&spnode_id, step);
    //             if(i < nParts){
    //                 build_supernode(i, tid, P[tid], S[tid]);
    //                 cnt++;
    //             }
    //         }
    //         LOG(INFO) << "tid=" << tid << " finish! cnt=" << cnt;
    //         }, thread_num
    //     );
    //     */
    // }

    // void build_supernode(idx_t partid, int tid, std::vector<vertex_t>& P, std::vector<vertex_t>& S){
    //     P.clear();
    //     S.clear();
    //     vertex_t source;
    //     auto inner_vertices = graph_->InnerVertices();
    //     double start = GetCurrentTime();
    //     for(auto u : inner_vertices){
    //         if(graph_part[u.GetValue()] == partid){
    //             // P.insert(u);
    //             P.emplace_back(u);
    //             source = u;
    //         }
    //     }
    //     LOG(INFO) << "tid=" << tid << " get_p time=" << (GetCurrentTime() - start);
    //     start = GetCurrentTime();
    //     for(auto d : P){
    //         const auto& ies = graph_->GetIncomingAdjList(d);
    //         for(auto& e : ies){
    //             if(graph_part[e.neighbor.GetValue()] != partid){
    //                 // S.insert(d);
    //                 S.emplace_back(d);
    //             }
    //         }
    //     }
    //     LOG(INFO) << "tid=" << tid << " get_s time=" << (GetCurrentTime() - start);
    //     LOG(INFO) << "s.size=" << S.size() << " P.size=" << P.size();
    //     start = GetCurrentTime();
    //     {
    //         /* create a set of indexes for each entry */
    //         if(S.size() == 0){
    //             // S.insert(source); // There is no entry point. At this time, the source point is used to form a super point.
    //             S.emplace_back(source);; // There is no entry point. At this time, the source point is used to form a super point.
    //         }
    //         for(auto src: S){
    //             Fc[src] = partid; // Ensure that sources are in the position of Fc[source][0]
    //         }
    //         int ids_id = partid;
    //         {
    //             // std::unique_lock<std::mutex> lk(supernode_ids_mux_);
    //             // supernode_ids.emplace_back(P.begin(), P.end());
    //             // ids_id = int(supernode_ids.size()) - 1;
    //             // supernode_bound_ids.emplace_back(O.begin(), O.end());
    //             supernode_ids[partid].insert(supernode_ids[partid].begin(), P.begin(), P.end());
    //             supernode_source[partid].insert(supernode_source[partid].begin(), S.begin(), S.end());
    //             // supernode_bound_ids[partid].insert(supernode_bound_ids.begin(), O.begin(), O.end());
    //         }
    //         for(auto u : P){
    //             Fc[u] = -(ids_id+1);
    //             id2spids[u] = ids_id;
    //         }
    //         for(auto src : S){
    //             Fc[src] = ids_id;
    //             /* build a supernode */
    //             // vid_t supernode_id = supernodes_num;
    //             // supernodes_num++;
    //             vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
    //             Fc_map[src] = supernode_id;
    //             supernodes[supernode_id].id = src;
    //             supernodes[supernode_id].ids = ids_id;
    //         }
    //     }
    //     LOG(INFO) << "tid=" << tid << " build time=" << (GetCurrentTime() - start);
    // }

    // big to small, sort
    static bool cmp_pair_b2s_cluster(std::pair<vid_t, vid_t> a, std::pair<vid_t, vid_t> b){
        if(a.second != b.second)
            return a.second > b.second;
        else return a.first < b.first;
    }

    // big to small, sort
    static bool cmp_pair_b2s(std::pair<vertex_t, vid_t> a, std::pair<vertex_t, vid_t> b){
        if(a.second != b.second)
            return a.second > b.second;
        else return a.first.GetValue() < b.first.GetValue();
    }

    // big to small, sort
    static bool cmp_pair_s2b_sort(std::pair<vertex_t, vid_t> a, std::pair<vertex_t, vid_t> b){
        if(a.second != b.second)
            return a.second < b.second;
        else return a.first.GetValue() < b.first.GetValue();
    }

    // small to big, priority_queue
    struct cmp_pair_s2b{
        bool operator()(std::pair<vertex_t, vid_t> a, std::pair<vertex_t, vid_t> b){
            return a.second < b.second;
        }
    };

    // void compress_by_metis(idx_t nParts){
    //     std::vector<std::set<vertex_t>> Parts;
    //     Parts.resize(nParts);
    //     for(auto u : graph_->InnerVertices()){
    //         Parts[graph_part[u.GetValue()]].insert(u);
    //     }
    //     vid_t supernodes_num = 0;
    //     vid_t cluster_num = 0;
    //     for(auto P : Parts){
    //         if(P.size() >= MIN_NODE_NUM){
    //             build_supernode_by_P(P);
    //         }
    //         cluster_num += 1;
    //         supernodes_num += P.size();
    //         if(cluster_num % 100000 == 0){
    //             LOG(INFO) << "cluster_num=" << cluster_num << " spnodes_num=" << supernodes_num << std::endl;
    //         }
    //     }
    //     LOG(INFO) << "cluster_num=" << cluster_num << " spnodes_num=" << supernodes_num << std::endl;
    // }

    /**
        road_usa.e.c:
        每一行是一个社区，每一行第一个数字是社区大小：
        # 社区大小 {包含顶点的id}
        2 0 1
        3 3 4 5
        2 6 7
    */
    void compress_by_scanpp(){
        // std::string path = "/mnt/data/nfs/yusong/code/ppSCAN/SCANVariants/scan_plus2/result_uk-2002_base.txt.c";
        std::string path = FLAGS_efile + ".c_" 
            + std::to_string(FLAGS_max_node_num); // road_usa.e -> road_usa.e.c.1000
        std::vector<std::string> keys{"_w.", "_ud.", "_w1.", ".random."};
        while (true) {
            bool changed = false;
            for(int i = 0; i < keys.size(); i++){
                std::string key = keys[i];
                std::size_t found = path.rfind(key);
                if (found!=std::string::npos) {
                    path.replace (found, key.length(), ".");
                    changed = true;
                }
            }
            if (changed == false) {
                break;
            }
        }

        LOG(INFO) << "load cluster result file... path=" << path;

        std::ifstream inFile(path);
        if(!inFile){
            LOG(INFO) << "open file failed. " << path;
            exit(0);
        }
        size_t size;
        vid_t v_oid, v_gid;
        vid_t cluster_num = 0;
        vertex_t u;
        auto vm_ptr = graph_->vm_ptr();
        fid_t fid = this->graph_->fid();
        while(inFile >> size){
            std::set<vertex_t> P;
            for(int i = 0; i < size; i++){
                inFile >> v_oid;
                CHECK(vm_ptr->GetGid(v_oid, v_gid));
                fid_t v_fid = vm_ptr->GetFidFromGid(v_gid);
                if (v_fid == fid) {
                    vertex_t u;
                    CHECK(this->graph_->Gid2Vertex(v_gid, u));
                    P.insert(u);
                }
            }
            cluster_num++;
            // debug
            // {
            //     LOG(INFO) << "cluster_num=" << cluster_num << " p.size=" << P.size() << std::endl;
            //     for(auto p : P){
            //         std::cout << p.GetValue() << " ";
            //     }
            //     std::cout << std::endl;
            //     LOG(INFO) << std::endl;
            //     if(cluster_num > 4){
            //         return ;
            //     }
            // }
            if(P.size() >= MIN_NODE_NUM){
                build_supernode_by_P(P);
            }
            if(cluster_num % 100000 == 0){
                LOG(INFO) << "cluster_num=" << cluster_num << " spnodes_num=" << supernodes_num << std::endl;
            }
        }
    }

    /**
        supernode: in_node_num * out_node_num < index_edge_num
            index_num < inner_edge_num
            index_num = in_node_num * out_node_num
    */
    void build_supernode_by_P(std::set<vertex_t>& P){
        std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
        // std::set<vertex_t> P; // inner node
        std::set<vertex_t> B; // belong to P, bound vertices
        const float obj_score = 1;
        const float ring_weight = 1;

        long long temp_ie_num = 0;
        vertex_t s;
        // inner edge, out bound edge
        for(auto v : P){
            const auto& oes = graph_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) != P.end()){
                    temp_ie_num++;
                }
            }
        }
        // inner bound node
        for(auto v : P){
            const auto& oes = graph_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) == P.end()){
                    B.insert(v);
                    break;
                }
            }
        }
        // source node
        for(auto v : P){
            const auto& oes = graph_->GetIncomingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) == P.end()){
                    S.insert(v);
                    break;
                }
            }
        }
        int b_num = B.size();
        int s_num = S.size();
        float score = temp_ie_num * 1.0 / (s_num * b_num + 1e-3); // must >= 0
        float obj = FLAGS_compress_threshold; // 1
        if(score >= obj){
        // if(1){ // not fiter
            if(S.size() == 0){
                S.insert(*(P.begin()));
            }
            int ids_id = -1;
            {
                std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                supernode_ids.emplace_back(P.begin(), P.end());
                ids_id = int(supernode_ids.size()) - 1; // root_id
                // supernode_bound_ids.emplace_back(O.begin(), O.end());
                supernode_source.emplace_back(S.begin(), S.end());
            }
            // CHECK(ids_id >= 0);
            for(auto u : P){
                Fc[u] = -(ids_id+1);
                id2spids[u] = ids_id;
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                // vid_t supernode_id = supernodes_num;
                // supernodes_num++;
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id; // root_id
            }
        }
    }


    /**
     * 解序列化,从文件中读取构建好的cluster
    */
    bool de_serialize_cluster(const std::string prefix, vid_t& init_mirror_num) {
      std::fstream file(prefix, std::ios::in | std::ios::binary);
      if(!file){
        LOG(INFO) << "Can't opening file, refind cluster... " << prefix;
        return false;
      }
      LOG(INFO) << "Deserializing cluster to " << prefix;
      // read cluster
      // vid_t init_mirror_num = 0;
      file.read(reinterpret_cast<char *>(&init_mirror_num), sizeof(vid_t));
      vid_t supernode_ids_num = 0;
      file.read(reinterpret_cast<char *>(&supernode_ids_num), sizeof(vid_t));

      LOG(INFO) << " init_mirror_num=" << init_mirror_num;
      LOG(INFO) << " supernode_ids_num=" << supernode_ids_num;

      // init 
      this->supernode_ids.resize(supernode_ids_num);
      this->cluster_ids.resize(supernode_ids_num);
      this->supernode_source.resize(supernode_ids_num);
      this->supernode_in_mirror.resize(supernode_ids_num);
      this->supernode_out_mirror.resize(supernode_ids_num);

      for (vid_t i = 0; i < supernode_ids_num; i++) {
        std::vector<vertex_t>& sp_ids = this->supernode_ids[i];
        vid_t sp_ids_num = 0;
        file.read(reinterpret_cast<char *>(&sp_ids_num), sizeof(vid_t));
        sp_ids.resize(sp_ids_num);
        for (vid_t i = 0; i < sp_ids_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_ids[i] = vertex_t(vid);
        }

        std::vector<vertex_t>& cs_ids = this->cluster_ids[i];
        vid_t cs_ids_num = 0;
        file.read(reinterpret_cast<char *>(&cs_ids_num), sizeof(vid_t));
        cs_ids.resize(cs_ids_num);
        for (vid_t i = 0; i < cs_ids_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          cs_ids[i] = vertex_t(vid);
        }

        std::vector<vertex_t>& sp_source = this->supernode_source[i];
        vid_t sp_source_num = 0;
        file.read(reinterpret_cast<char *>(&sp_source_num), sizeof(vid_t));
        sp_source.resize(sp_source_num);
        for (vid_t i = 0; i < sp_source_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_source[i] = vertex_t(vid);
        }

        std::unordered_set<vertex_t>& sp_in_mirror = this->supernode_in_mirror[i];
        vid_t sp_in_mirror_num = 0;
        file.read(reinterpret_cast<char *>(&sp_in_mirror_num), sizeof(vid_t));
        for (vid_t i = 0; i < sp_in_mirror_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_in_mirror.insert(vertex_t(vid));
        }

        std::unordered_set<vertex_t>& sp_out_mirror = this->supernode_out_mirror[i];
        vid_t sp_out_mirror_num = 0;
        file.read(reinterpret_cast<char *>(&sp_out_mirror_num), sizeof(vid_t));
        for (vid_t i = 0; i < sp_out_mirror_num; i++) {
          vid_t vid = 0;
          file.read(reinterpret_cast<char *>(&vid), sizeof(vid_t));
          sp_out_mirror.insert(vertex_t(vid));
        }
      }

      fc_t id = 0;
      for(auto v : graph_->Vertices()){
        file.read(reinterpret_cast<char *>(&id), sizeof(fc_t));
        Fc[v] = id;
      }

      vid_t spid;
      for(auto v : graph_->Vertices()){
        file.read(reinterpret_cast<char *>(&spid), sizeof(vid_t));
        id2spids[v] = spid;
      }
      int eof = 0;
      file.read(reinterpret_cast<char *>(&eof), sizeof(int));
      file.close();
      CHECK_EQ(eof, -1);
      return true;
    }

    /**
     * 解序列化,从文件中读取构建好的cluster
    */
    void serialize_cluster(const std::string prefix, vid_t init_mirror_num) {
      std::fstream file(prefix, std::ios::out | std::ios::binary);
      if(!file){
        LOG(INFO) << "Error opening file. " << prefix;
        exit(0);
      }
      LOG(INFO) << "Serializing cluster to " << prefix;
      // write cluster
      file.write(reinterpret_cast<char *>(&init_mirror_num), sizeof(vid_t));
      vid_t supernode_ids_num = supernode_ids.size();
      file.write(reinterpret_cast<char *>(&supernode_ids_num), sizeof(vid_t));

      LOG(INFO) << " init_mirror_num=" << init_mirror_num;
      LOG(INFO) << " supernode_ids_num=" << supernode_ids_num;

      for (vid_t i = 0; i < supernode_ids_num; i++) {
        std::vector<vertex_t>& sp_ids = this->supernode_ids[i];
        vid_t sp_ids_num = sp_ids.size();
        file.write(reinterpret_cast<char *>(&sp_ids_num), sizeof(vid_t));
        for (auto& v : sp_ids) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::vector<vertex_t>& cs_ids = this->cluster_ids[i];
        vid_t cs_ids_num = cs_ids.size();
        file.write(reinterpret_cast<char *>(&cs_ids_num), sizeof(vid_t));
        for (auto& v : cs_ids) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::vector<vertex_t>& sp_source = supernode_source[i];
        vid_t sp_source_num = sp_source.size();
        file.write(reinterpret_cast<char *>(&sp_source_num), sizeof(vid_t));
        for (auto& v : sp_source) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::unordered_set<vertex_t>& sp_in_mirror = supernode_in_mirror[i];
        vid_t sp_in_mirror_num = sp_in_mirror.size();
        file.write(reinterpret_cast<char *>(&sp_in_mirror_num), sizeof(vid_t));
        for (auto& v : sp_in_mirror) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }

        std::unordered_set<vertex_t>& sp_out_mirror = supernode_out_mirror[i];
        vid_t sp_out_mirror_num = sp_out_mirror.size();
        file.write(reinterpret_cast<char *>(&sp_out_mirror_num), sizeof(vid_t));
        for (auto& v : sp_out_mirror) {
          vid_t id = v.GetValue();
          file.write(reinterpret_cast<char *>(&id), sizeof(vid_t));
        }
      }
      fc_t id;
      for (auto fc : Fc) {
          id = fc;
          file.write(reinterpret_cast<char *>(&id), sizeof(fc_t));
      }

      vid_t spid;
      for (auto spid : id2spids) {
          file.write(reinterpret_cast<char *>(&spid), sizeof(vid_t));
      }
      int eof = -1;
      file.write(reinterpret_cast<char *>(&eof), sizeof(int));
      file.close();
    }

    /**
        road_usa.e.c_5000:
        每一行是一个社区，每一行第一个数字是社区大小：
        # 社区大小 {包含顶点的id}
        2 0 1
        3 3 4 5
        2 6 7
    */
    void compress_by_cluster(const std::string prefix){
        LOG(INFO) << "compress_by_cluster...";

        if (prefix != "") {
          vid_t init_mirror_num = 0;
          bool find = de_serialize_cluster(prefix, init_mirror_num);
          if (find == true) {
            final_build_supernode(init_mirror_num);
            return ;
          }
        }

        // std::string path = "/mnt/data/nfs/yusong/code/ppSCAN/SCANVariants/scan_plus2/result_uk-2002_base.txt.c";
        // get cluster file name
        std::string path = FLAGS_efile + ".c_" 
            + std::to_string(FLAGS_max_node_num); // road_usa.e -> road_usa.e.c.1000
        std::vector<std::string> keys{"_w.", "_ud.", "_w1.", ".random."};
        while (true) {
            bool changed = false;
            for(int i = 0; i < keys.size(); i++){
                std::string key = keys[i];
                std::size_t found = path.rfind(key);
                if (found!=std::string::npos) {
                    path.replace (found, key.length(), ".");
                    changed = true;
                }
            }
            if (changed == false) {
                break;
            }
        }

        LOG(INFO) << "load cluster result file... path=" << path;

        /* read cluster file */
        std::ifstream inFile(path);
        if(!inFile){
            LOG(INFO) << "open file failed. " << path;
            exit(0);
        }
        size_t size;
        vid_t v_oid, v_gid;
        vid_t cluster_num = 0;
        vertex_t u;
        auto vm_ptr = graph_->vm_ptr();
        VertexArray<vid_t, vid_t> id2clusterid; // map: vid -> clusterid
        id2clusterid.Init(graph_->Vertices(), ID_default_value);
        std::vector<std::vector<vertex_t> > clusters;
        fid_t fid = this->graph_->fid();
        vid_t max_v_id = graph_->GetVerticesNum();
        vid_t load_cnt = 0;
        while(inFile >> size){
            std::set<vertex_t> P;
            for(int i = 0; i < size; i++){
                inFile >> v_oid;
                CHECK_GE(max_v_id, v_oid);
                CHECK(vm_ptr->GetGid(v_oid, v_gid));
                fid_t v_fid = vm_ptr->GetFidFromGid(v_gid);
                if (v_fid == fid) {
                    vertex_t u;
                    CHECK(this->graph_->Gid2Vertex(v_gid, u));
                    // LOG(INFO) << v_oid << " " << v_gid << " " << u.GetValue();
                    P.insert(u);
                    id2clusterid[u] = cluster_num;
                    // for (auto e : graph_->GetOutgoingAdjList(u)) {
                    //     LOG(INFO) << "v_gid=" << v_gid << " " << e.neighbor.GetValue();
                    // }
                }
            }
            if(P.size() >= MIN_NODE_NUM){
              clusters.emplace_back(P.begin(), P.end());
              cluster_num++;
                // build_supernode_by_P(P);
            }
            if(load_cnt % 100000 == 0){
                LOG(INFO) << "load_cnt=" << load_cnt 
                          << " cluster_num=" << cluster_num << std::endl;
            }
            load_cnt++;
        }
        LOG(INFO) << "cluster_num=" << cluster_num 
                  << " clusters.size=" << clusters.size();

        vid_t init_mirror_num = get_init_supernode_by_clusters(clusters, 
                                                                id2clusterid);
        /* 将cluster序列化到文件: 注意必须放在build_supernode之前 */
        if (prefix != "") {
          serialize_cluster(prefix, init_mirror_num);
        }
        final_build_supernode(init_mirror_num);
    }

    /* 通过cluster建立超点，必要的地方添加mirror点 */
    vid_t get_init_supernode_by_clusters (std::vector<std::vector<vertex_t>> 
                                        &clusters, VertexArray<vid_t, vid_t> 
                                        &id2clusterid) {
        LOG(INFO) << "build supernode by out/in-mirror...";
        double init_supernode_by_clusters_time = GetCurrentTime();
        // debug  (入口点+出口点)统计如果采用Mirror-Master能对边的减少率能提高多少, 
        // 前期没有入口*出口过滤，在此处过滤
        typedef long long count_t;
        count_t k = FLAGS_mirror_k; // 阈值
        count_t mirror_num = 0;
        count_t reduce_edge_num = 0;
        count_t new_index_num = 0;
        count_t old_index_num = 0;
        count_t old_inner_edge = 0;
        count_t new_inner_edge = 0;
        count_t spnids_num = 0;
        count_t add_spnids_num = 0;
        count_t all_old_exit_node_num = 0;
        count_t all_new_exit_node_num = 0;
        count_t all_old_entry_node_num = 0;
        count_t all_new_entry_node_num = 0;
        count_t abandon_node_num = 0; // 不满足入口*出口舍弃的点
        count_t abandon_edge_num = 0;
        const vid_t spn_ids_num = clusters.size(); 
        float obj = FLAGS_compress_threshold; // 1
        count_t no_a_entry_node_num = 0;
        count_t mirror_node_num = 0;
        LOG(INFO) << " FLAGS_compress_threshold=" << FLAGS_compress_threshold;

        /* 按照内部顶点数量从大到小排序 */
        // double sort_time = GetCurrentTime();
        // std::vector<std::pair<vid_t, vid_t>> init_cluster;
        // init_cluster.reserve(spn_ids_num);
        // for(int j = 0; j < spn_ids_num; j++){
        //     init_cluster.emplace_back(
        //                     std::pair<vid_t, vid_t>(j, clusters[j].size()));
        // }
        // std::sort(init_cluster.begin(), init_cluster.end(), cmp_pair_b2s_cluster); // big to small
        // LOG(INFO) << "  sort_time=" << (GetCurrentTime() - sort_time);
        // LOG(INFO) << "    first=" << init_cluster[0].first 
        //           << ", " << init_cluster[0].second;
        // LOG(INFO) << "    last=" << init_cluster[spn_ids_num-1].first 
        //           << ", " << init_cluster[spn_ids_num-1].second;

        //------------------------------------------------------------------
        // 为了将不加Mirrror点的功能融合，这里假设如果Mirror点最大值设置为1e8
        //   时，默认为不需要统计mirror点的相关信息，从不必浪费统计时间
        //------------------------------------------------------------------
        bool is_use_mirror = true;
        if (FLAGS_mirror_k == 1e8) {
          is_use_mirror = false;
          LOG(INFO) << "Close function of Using Mirror!";
        } else {
          LOG(INFO) << "Open function of Using Mirror!";
        }
        
        // for (vid_t cid = 0; cid < spn_ids_num; cid++){
        //     vid_t j = init_cluster[cid].first;
	unsigned long int number = 0;
        for (vid_t j = 0; j < spn_ids_num; j++) {
            std::vector<vertex_t> &node_set = clusters[j];
            // 统计所有入口点/出口点的源顶点
            std::unordered_map<vid_t, vid_t> in_frequent;
            std::unordered_map<vid_t, vid_t> out_frequent;
            std::set<vertex_t> old_entry_node;
            std::set<vertex_t> old_exit_node;
            count_t temp_old_inner_edge = 0;
            for (auto u : node_set) {
		
		if(number % 100000 == 0)
              LOG(INFO) << "u is "<< number;
		number++;
              for (auto e : this->graph_->GetIncomingAdjList(u)) {
                vid_t to_ids = id2clusterid[e.neighbor];
                if (to_ids != j) { // 外部点
                  if (is_use_mirror == true) {
                    vid_t newids = id2spids[e.neighbor];
                    if (newids != ID_default_value) { // new cluster
                      auto out_mirror = supernode_out_mirror[newids];
                      if (out_mirror.find(u) == out_mirror.end()) {
                        // 外部点，因为u不在入邻居所在cluster的out-mirror中
                        in_frequent[e.neighbor.GetValue()] += 1;
                      } else {
                        // 在out-mirror中，则neighbor->u的边应该被u'->u的同步边替换
                      }
                    } else {
                      // 外部点
                      in_frequent[e.neighbor.GetValue()] += 1;
                    }
                  }
                  old_entry_node.insert(u);
                } else {
                    temp_old_inner_edge++;
                }
              }
              for (auto e : this->graph_->GetOutgoingAdjList(u)) {
                vid_t to_ids = id2clusterid[e.neighbor];
                if (to_ids != j) { // 外部点
                  if (is_use_mirror == true) {
                    vid_t newids = id2spids[e.neighbor];
                    if (newids != ID_default_value) { // new cluster
                      auto in_mirror = supernode_in_mirror[newids];
                      if (in_mirror.find(u) == in_mirror.end()) {
                        // 外部点，因为u不在出邻居所在cluster的in-mirror中
                        out_frequent[e.neighbor.GetValue()] += 1;
                      } else {
                          // 在in-mirror中，则u->neighbor的边应该被u->u'的同步边替换
                      }
                    } else {
                      // 外部点
                      out_frequent[e.neighbor.GetValue()] += 1;
                    }
                  }
                  old_exit_node.insert(u);
                }
              }
            }
            // 分析出现频率
            count_t in_edge_num = 0;
            count_t out_edge_num = 0;
            count_t in_mirror_node_num = 0;
            count_t out_mirror_node_num = 0;
            count_t old_exit_node_num = old_exit_node.size();
            count_t old_entry_node_num = old_entry_node.size();
            std::set<vertex_t> old_P;
            old_P.insert(node_set.begin(), node_set.end());
            std::set<vertex_t> in_P;
            in_P.insert(node_set.begin(), node_set.end());
            std::set<vertex_t> out_P;
            out_P.insert(node_set.begin(), node_set.end());
            std::set<vertex_t> in_mirror;
            std::set<vertex_t> out_mirror;
            
            if (is_use_mirror == true) {
              for (const auto& fre : in_frequent) {
                  if (fre.second > k) {
                      in_edge_num += fre.second;
                      in_mirror_node_num += 1;
                      in_P.insert(vertex_t(fre.first));  // 高频入口点的源顶点作为Mirror点
                      in_mirror.insert(vertex_t(fre.first));  // 高频入口点的源顶点作为Mirror点
                  }
                  // LOG(INFO) << "in: " << fre.first << ": " << fre.second << std::endl;
              }
              for (const auto& fre : out_frequent) {
                  if (fre.second > k) {
                      out_edge_num += fre.second;
                      out_mirror_node_num += 1;
                      out_P.insert(vertex_t(fre.first));  // 高频出口点的源顶点作为Mirror点
                      out_mirror.insert(vertex_t(fre.first));  // 高频出口点的源顶点作为Mirror点
                  }
                  // LOG(INFO) << "out: " << fre.first << ": " << fre.second << std::endl;
              }
            }

            // 统计新的出口点
            std::set<vertex_t> B; // belong to P, bound vertices
            for(auto v : node_set){ // 遍历原来的点集
            // parallel_for(int i = 0; i < node_set.size(); i++){
            //   vertex_t v = node_set[i];
              const auto& oes = this->graph_->GetOutgoingAdjList(v);
              for(auto& e : oes){
                if(out_P.find(e.neighbor) == out_P.end()){ // 包含Mirror
                  {
                    // std::unique_lock<std::mutex> lk(set_mux_);
                    B.insert(v);
                  }
                  break;
                }
              }
            }
            // 统计新的入口点
            std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
            for(auto v : node_set){ // 遍历原来的点集
            // parallel_for(int i = 0; i < node_set.size(); i++){
            //   vertex_t v = node_set[i];
              const auto& oes = this->graph_->GetIncomingAdjList(v);
              for(auto& e : oes){
                if(in_P.find(e.neighbor) == in_P.end()){ // 包含Mirror
                  {
                    // std::unique_lock<std::mutex> lk(set_mux_);
                    S.insert(v);
                  }
                  break;
                }
              }
            }
            // 统计最终结果
            count_t new_exit_node_num = B.size() + out_mirror_node_num;
            count_t new_entry_node_num = S.size() + in_mirror_node_num;
            count_t temp_old_index_num = old_exit_node_num * old_entry_node_num;
            count_t temp_entry_index_num = old_exit_node_num * new_entry_node_num;
            count_t temp_exit_index_num = new_exit_node_num * old_entry_node_num;
            count_t temp_new_index_num = new_exit_node_num * new_entry_node_num;

            const bool original_compress_condition = 
                (temp_old_index_num < temp_old_inner_edge);
            // const bool mirror_compress_condition = 
            //   (temp_new_index_num + in_mirror_node_num + out_mirror_node_num
            //     < temp_old_inner_edge + in_edge_num + out_edge_num); // 加Mirror后是否应该压缩

            std::vector<float> benefit;
            benefit.resize(4, 0);
            // 需要考虑分母为0的情况：benef
            // benefit[0] = temp_old_inner_edge * 1.0 / temp_old_index_num; // 不加mirror
            // benefit[1] = (temp_old_inner_edge + in_edge_num) * 1.0 
            //     / (temp_entry_index_num + in_mirror_node_num); // 加入口点mirror
            // benefit[2] = (temp_old_inner_edge + out_edge_num) * 1.0
            //     / (temp_exit_index_num + out_mirror_node_num); // 加出口点mirror
            // benefit[3] = (temp_old_inner_edge + in_edge_num + out_edge_num) * 1.0
            //     / (temp_new_index_num + in_mirror_node_num + out_mirror_node_num); // 入+出miiror
            benefit[0] = temp_old_inner_edge * 1.0 - temp_old_index_num; // 不加mirror
            benefit[1] = (temp_old_inner_edge + in_edge_num) * 1.0 
                         - (temp_entry_index_num + in_mirror_node_num); // 加入口点mirror
            benefit[2] = (temp_old_inner_edge + out_edge_num) * 1.0
                         - (temp_exit_index_num + out_mirror_node_num); // 加出口点mirror
            benefit[3] = (temp_old_inner_edge + in_edge_num + out_edge_num) * 1.0
                         - (temp_new_index_num + in_mirror_node_num + out_mirror_node_num); // 入+出miiror

            int max_i = 0;
            for (int i = 0; i < benefit.size(); i++) {
                // LOG(INFO) << "benefit[" << i << "]=" << benefit[i];
                if (benefit[max_i] < benefit[i]) {
                    max_i = i;
                }
            }
            float max_benefit = benefit[max_i];
            // LOG(INFO) << "== max_i=" << max_i << " max_benefit=" << max_benefit;

            // 统计未能压缩的点和边，即放弃的cluster
            if (max_benefit <= obj) {
                abandon_edge_num += temp_old_inner_edge;
                abandon_node_num += old_P.size();
            }

            // 不加mirror点的情况
            if (original_compress_condition == true) {
                spnids_num++;
                old_inner_edge += temp_old_inner_edge;
                old_index_num += temp_old_index_num;
                all_old_entry_node_num += old_entry_node_num;
                all_old_exit_node_num += old_exit_node_num;
            }
            // 四种方案中选择一种
            // LOG(INFO) << "--obj=" << obj;
            // LOG(INFO) << "--max_benefit=" << max_benefit;
            // LOG(INFO) << "--max_i=" << max_i;
            if (max_benefit > obj) {
                if (original_compress_condition == false) { // 未加Mirror时未压缩
                    add_spnids_num++; // 仅仅因为加mirror才成为cluster
                } 
                new_inner_edge += temp_old_inner_edge;
                if (max_i == 0) { // 不加mirror
                    new_index_num += temp_old_index_num;
                    all_new_entry_node_num += old_entry_node_num;
                    all_new_exit_node_num += old_exit_node_num;
                } else if (max_i == 1) { // 加入口点mirror
                    mirror_num += in_mirror_node_num;
                    reduce_edge_num += in_edge_num;
                    new_index_num += temp_entry_index_num;
                    all_new_entry_node_num += new_entry_node_num;
                    all_new_exit_node_num += old_exit_node_num;
                } else if (max_i == 2) { // 加出口点mirror
                    mirror_num += out_mirror_node_num;
                    reduce_edge_num += out_edge_num;
                    new_index_num += temp_exit_index_num;
                    all_new_entry_node_num += old_entry_node_num;
                    all_new_exit_node_num += new_exit_node_num;
                } else if (max_i == 3) {
                    mirror_num += in_mirror_node_num;
                    mirror_num += out_mirror_node_num;
                    reduce_edge_num += in_edge_num;
                    reduce_edge_num += out_edge_num;
                    new_index_num += temp_new_index_num;
                    all_new_entry_node_num += new_entry_node_num;
                    all_new_exit_node_num += new_exit_node_num;
                } else {
                    LOG(INFO) << "no this type. max_i=" << max_i;
                    exit(0);
                }

                // get new S, in_mirror, out_mirror
                if (max_i == 0) {
                    S = old_entry_node;
                    in_mirror.clear();
                    out_mirror.clear();
                } else if (max_i == 1) {
                    out_mirror.clear();
                } else if (max_i == 2) {
                    S = old_entry_node;
                    in_mirror.clear();
                }

                // build supernode:
                // if(S.size() == 0 && in_mirror.size() == 0){
                //     S.insert(*(old_P.begin())); // is it necessary? // 好像没必要,
                //     // no_a_entry_node_num++;
                //     __sync_fetch_and_add(&no_a_entry_node_num, 1);
                //     // LOG(INFO) << "no a entry node in this cluster.";
                // }
                int ids_id = -1;
                {
                    std::unique_lock<std::mutex> lk(supernode_ids_mux_);
                    supernode_ids.emplace_back(old_P.begin(), old_P.end());
                    ids_id = int(supernode_ids.size()) - 1; // root_id
                    cluster_ids.emplace_back(old_P.begin(), old_P.end());
                    supernode_source.emplace_back(S.begin(), S.end());
                    supernode_in_mirror.emplace_back(in_mirror.begin(), 
                                                     in_mirror.end());
                    supernode_out_mirror.emplace_back(out_mirror.begin(), 
                                                      out_mirror.end());
                    mirror_node_num += in_mirror.size();
                    mirror_node_num += out_mirror.size();
                }
                // CHECK(ids_id >= 0);
                for(auto u : old_P){
                    Fc[u] = -(ids_id+1);
                    id2spids[u] = ids_id;
                }
            }
        }
        

        LOG(INFO) << "  init_supernode_by_clusters_time=" 
                  << (GetCurrentTime() - init_supernode_by_clusters_time);

        // print_cluster();

        LOG(INFO) << "----------------------------------------";
        LOG(INFO) << "  init mirror_node_num=" << mirror_node_num;
        LOG(INFO) << " no_a_entry_node_num=" << no_a_entry_node_num;
        size_t local_edge_num = graph_->GetEdgeNum() / 2;
        float new_edge_reduce_rato = (new_inner_edge + reduce_edge_num * 1.0
                                   - mirror_num - new_index_num) 
                                   / local_edge_num;
        float old_edge_reduce_rato = (old_inner_edge - old_index_num) * 1.0
                                   / local_edge_num;
        LOG(INFO) << " new_edge_reduce_rato=" << new_edge_reduce_rato;
        LOG(INFO) << " old_edge_reduce_rato=" << old_edge_reduce_rato;
        LOG(INFO) << " finish build supernode... supernodes_num=" 
                  << supernodes_num;
        LOG(INFO) << " old_node_num=" << graph_->GetVerticesNum() 
                  << " all_edge_num=" << local_edge_num;
        LOG(INFO) << "-----------------------------------------";
        LOG(INFO) << "k=" << k << " mirror_num= " << mirror_num 
                  << " reduce_edge_num=" << reduce_edge_num 
                  << " new_index_num=" << new_index_num
                  << " old_index_num=" << old_index_num;
        LOG(INFO) << "spnids_num=" << spnids_num 
                  << " add_spnids_num=" << add_spnids_num
                  << " new_inner_edge=" << new_inner_edge
                  << " old_inner_edge=" << old_inner_edge;
        LOG(INFO) << "all_old_entry_node_num=" << all_old_entry_node_num
                  << " all_old_exit_node_num=" << all_old_exit_node_num
                  << " all_new_entry_node_num=" << all_new_entry_node_num
                  << " all_new_exit_node_num=" << all_new_exit_node_num;
        LOG(INFO) << "abandon_edge_num=" << abandon_edge_num 
                  << " abandon_node_num=" << abandon_node_num;
      return mirror_node_num;
    }

    /**
     * 根据划分的cluster以及mirror点, 为每个入口点建立shortcut.
    */
    void final_build_supernode(vid_t mirror_node_num) {
        LOG(INFO) << "  --------------build_supernode----------------------";
        double final_build_supernode = GetCurrentTime();
        size_t mirror_node_cnt = 0;
        size_t inmirror2source_cnt = 0;
        size_t outmirror2source_cnt = 0;
        auto new_node_range = VertexRange<vid_t>(0, 
                                              old_node_num + mirror_node_num);
        Fc_map.Init(new_node_range, ID_default_value);
        for (vid_t i = 0; i < supernode_ids.size(); i++) {
            vid_t ids_id = i;
            auto& old_P = supernode_ids[ids_id];
            auto& S = supernode_source[ids_id];
            auto& in_mirror = supernode_in_mirror[ids_id];
            auto& out_mirror = supernode_out_mirror[ids_id];
            /* 针对外部点被当作mirror的情况进行处理 */
            std::vector<vertex_t> del_v;
            for(auto v : supernode_in_mirror[i]) {
                if (Fc[v] == FC_default_value) {
                    mirror_node_cnt++;
                    old_P.emplace_back(v);
                    cluster_ids[ids_id].emplace_back(v);
                    // supernode_in_mirror[ids_id].erase(v);
                    del_v.emplace_back(v);
                    const auto& ies = this->graph_->GetIncomingAdjList(v);
                    for(auto& e : ies){
                        if(id2spids[e.neighbor] != ids_id 
                        //    ){ // new source
                           && in_mirror.find(e.neighbor) == in_mirror.end()){ // new source // 测试测试！！！！！！！！！！！！
                            S.emplace_back(v);
                            inmirror2source_cnt++;
                            break;
                        }
                    }
                    Fc[v] = -(ids_id+1);
                    id2spids[v] = ids_id;
                }
            }
            for (auto v : del_v) {
                supernode_in_mirror[ids_id].erase(v);
                if (out_mirror.find(v) != out_mirror.end()) { // 针对同时是out-mirror的情况
                    out_mirror.erase(v);
                }
            }
            del_v.clear();
            for(auto v : supernode_out_mirror[i]) {
                if (Fc[v] == FC_default_value) {
                    mirror_node_cnt++;
                    old_P.emplace_back(v);
                    cluster_ids[ids_id].emplace_back(v);
                    // supernode_out_mirror[ids_id].erase(v);
                    del_v.emplace_back(v);
                    const auto& ies = this->graph_->GetIncomingAdjList(v);
                    for(auto& e : ies){
                        if(id2spids[e.neighbor] != ids_id 
                            // ){ // new source
                           && in_mirror.find(e.neighbor) == in_mirror.end()){ // new source // 测试测试！！！！！！
                            S.emplace_back(v);
                            outmirror2source_cnt++;
                            break;
                        }
                    }
                    Fc[v] = -(ids_id+1);
                    id2spids[v] = ids_id;
                }
            }
            for (auto v : del_v) {
                supernode_out_mirror[ids_id].erase(v);
                if (in_mirror.find(v) != in_mirror.end()) {
                    in_mirror.erase(v);
                }
            }
            // 下面的代码为啥每次扩展1空间？这样应该很慢！！！
            cluster_in_mirror_ids.resize(cluster_in_mirror_ids.size() + 1);
            cluster_out_mirror_ids.resize(cluster_out_mirror_ids.size() + 1);
            for (auto v : in_mirror) {
                vid2in_mirror_cluster_ids[v.GetValue()].emplace_back(ids_id);
                // vid_t mirror_id = __sync_fetch_and_add(&all_node_num, 1);
                vid_t mirror_id = all_node_num++; // have a lock
                vertex_t m(mirror_id);
                mirrorid2vid[m] = v;
                // vid2mirrorid[v] = m;
                cluster_ids[ids_id].emplace_back(m);
                cluster_in_mirror_ids[ids_id].emplace_back(m);
            }
            for (auto v : out_mirror) {
                // out_mirror2spids[v.GetValue()].emplace_back(ids_id);
                // vid_t mirror_id = __sync_fetch_and_add(&all_node_num, 1);
                vid_t mirror_id = all_node_num++; // have a lock
                vertex_t m(mirror_id);
                mirrorid2vid[m] = v;
                // vid2mirrorid[v] = m;
                cluster_ids[ids_id].emplace_back(m);
                cluster_out_mirror_ids[ids_id].emplace_back(m);
            }
            for(auto src : S){
                Fc[src] = ids_id;
                /* build a supernode */
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                Fc_map[src] = supernode_id;
                shortcuts[src.GetValue()][ids_id] = supernode_id; // 线程不安全
                supernodes[supernode_id].id = src;
                supernodes[supernode_id].ids = ids_id; // root_id
            }
            for (auto mid : cluster_in_mirror_ids[ids_id]) {
                vid_t supernode_id = __sync_fetch_and_add(&supernodes_num, 1);
                // vid2in_mirror_cluster_ids[v.GetValue()].emplace_back(ids_id);
                vid2in_mirror_mids[mirrorid2vid[mid].GetValue()].emplace_back(
                                                                mid.GetValue());
                Fc_map[mid] = supernode_id;
                auto src = mirrorid2vid[mid];
                shortcuts[src.GetValue()][ids_id] = supernode_id; // 线程不安全 
                // supernodes[supernode_id].id = src;
                supernodes[supernode_id].id = mid; // source
                supernodes[supernode_id].ids = ids_id; // root_id
                // LOG(INFO) << "======2oid=" << this->graph_->GetId(src);
            }
            // get vertex's mirror address
            for (auto mid : cluster_out_mirror_ids[ids_id]) {
                vid2out_mirror_mids[mirrorid2vid[mid].GetValue()].emplace_back(
                                                                mid.GetValue());
            } 
        }

        // Fc_map.Resize(new_node_range);
        VertexArray<vid_t, vid_t> new_id2spids;
        new_id2spids.Init(new_node_range, ID_default_value);
        parallel_for(vid_t i = 0; i < old_node_num; i++) {
            vertex_t v(i);
            new_id2spids[v] = id2spids[v];
        }
        parallel_for(vid_t i = 0; i < this->cluster_ids.size(); i++) {
            for(auto v : this->cluster_in_mirror_ids[i]) {
                new_id2spids[v] = i;
            }
            for(auto v : this->cluster_out_mirror_ids[i]) {
                new_id2spids[v] = i;
            }
        }
        id2spids.Init(new_node_range);
        for (vertex_t v : new_node_range) {
            id2spids[v] = new_id2spids[v];
        }

        for(vid_t j = 0; j < this->cluster_ids.size(); j++){
          for (auto u : this->cluster_out_mirror_ids[j]) {
            this->all_out_mirror.emplace_back(u);
          }
        }

        this->indegree.resize(this->cluster_ids.size()+1);
        parallel_for (vid_t i = 0; i < this->cluster_ids.size(); i++) {
          vid_t sum = 0;
          for (auto v : this->supernode_ids[i]) {
            sum += this->graph_->GetIncomingAdjList(v).Size();
          }
          this->indegree[i] = sum;
        }
        this->indegree[this->GetClusterSize()] = this->graph_->GetEdgeNum() / 2;

        LOG(INFO) << "  get_initial_cluster_time="
                  << (GetCurrentTime() - final_build_supernode);
        LOG(INFO) << "  old_node_num=" << this->old_node_num; 
        LOG(INFO) << "  all_node_num=" << this->all_node_num; 
        LOG(INFO) << "out node in mirror: mirror_node_cnt=" << mirror_node_cnt;
        LOG(INFO) << "  inmirror2source_cnt =" << inmirror2source_cnt;
        LOG(INFO) << "  outmirror2source_cnt=" << outmirror2source_cnt;
    }

    /* 各个cluster组的子图，包含Mirror点 */
    void build_subgraph_mirror(const std::shared_ptr<fragment_t>& new_graph) {
        LOG(INFO) << "build_subgraph_mirror...";
        //subgraph
        double subgraph_time = GetCurrentTime();
        const vid_t spn_ids_num = this->supernode_ids.size();
        vid_t inner_node_num = this->all_node_num;
        subgraph.resize(inner_node_num);
        // std::vector<size_t> ia_oe_degree(inner_node_num+1, 0);
        vid_t ia_oe_num = 0;  
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
            std::vector<vertex_t> &node_set = this->supernode_ids[i];
            std::vector<vertex_t> &in_mirror_ids 
                                    = this->cluster_in_mirror_ids[i];
            std::vector<vertex_t> &out_mirror_ids 
                                    = this->cluster_out_mirror_ids[i];
            vid_t temp_a = 0;
            // auto ids_id = this->id2spids[*(node_set.begin())];
            // CHECK_EQ(ids_id, i);
            auto ids_id = i;
            for(auto v : node_set){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        subgraph[v.GetValue()].emplace_back(oe);
                    }
                }
            }
            for(auto m_id : in_mirror_ids){
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_out_mirror = 
                    this->supernode_out_mirror[v_superid];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    // 入口优先，不需要过滤掉存在出口Mirror的点
                    if(this->id2spids[oe.neighbor] == ids_id
                        ){ // in-mirror edge
                        // && v_out_mirror.find(oe.neighbor) == v_out_mirror.end()){ // in-mirror edge
                        subgraph[m_id.GetValue()].emplace_back(nbr_t(oe.neighbor,
                                                                    oe.data));
                    }
                }
            }
            for(auto m_id : out_mirror_ids){
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_in_mirror = 
                    this->supernode_in_mirror[v_superid]; 
                const auto& ies = new_graph->GetIncomingAdjList(v);
                for(auto& ie : ies){
                    // 入口优先，滤掉存在入口的Mirror的点
                    if(this->id2spids[ie.neighbor] == ids_id
                        //  ){ // out-mirror edge
                        && v_in_mirror.find(ie.neighbor) == v_in_mirror.end()){ // out-mirror edge
                        subgraph[ie.neighbor.GetValue()].emplace_back(nbr_t(m_id,
                                                                      ie.data));
                    }
                }
            }
        }

        double copy_subgraph_time = GetCurrentTime();
        this->subgraph_old = this->subgraph; // use to update.
        LOG(INFO) << "  copy_subgraph_time=" << (GetCurrentTime()-copy_subgraph_time);

        LOG(INFO) << "subgraph_time=" << (GetCurrentTime()-subgraph_time);

        // print_subgraph();
    }

    /* 各个cluster组的子图，包含Mirror点 */
    void inc_build_subgraph_mirror(const std::shared_ptr<fragment_t>& new_graph) {
        LOG(INFO) << "inc_build_subgraph_mirror...";
        //subgraph
        double inc_subgraph_time = GetCurrentTime();
        const vid_t spn_ids_num = this->update_cluster_ids.size();
        // vid_t inner_node_num = this->all_node_num;
        // subgraph.resize(inner_node_num); // 假设没有新增点
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
            auto ids_id = this->update_cluster_ids[i];
            std::vector<vertex_t> &node_set = this->supernode_ids[ids_id];
            std::vector<vertex_t> &in_mirror_ids 
                                    = this->cluster_in_mirror_ids[ids_id];
            std::vector<vertex_t> &out_mirror_ids 
                                    = this->cluster_out_mirror_ids[ids_id];
            vid_t temp_a = 0;
            // auto ids_id = this->id2spids[*(node_set.begin())];
            // CHECK_EQ(ids_id, i);
            for(auto v : node_set){
                subgraph[v.GetValue()].clear();
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        subgraph[v.GetValue()].emplace_back(oe);
                    }
                }
            }
            for(auto m_id : in_mirror_ids){
                subgraph[m_id.GetValue()].clear();
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_out_mirror = 
                    this->supernode_out_mirror[v_superid];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    // 入口优先，不需要过滤掉存在出口Mirror的点
                    if(this->id2spids[oe.neighbor] == ids_id
                         ){ // in-mirror edge
                        // && v_out_mirror.find(oe.neighbor) == v_out_mirror.end()){ // in-mirror edge
                        subgraph[m_id.GetValue()].emplace_back(nbr_t(oe.neighbor,
                                                                     oe.data));
                    }
                }
            }
            for(auto m_id : out_mirror_ids){
                vertex_t v = mirrorid2vid[m_id];
                auto v_superid = this->id2spids[v];
                std::unordered_set<vertex_t>& v_in_mirror = 
                    this->supernode_in_mirror[v_superid]; 
                const auto& ies = new_graph->GetIncomingAdjList(v);
                for(auto& ie : ies){
                    // 入口优先，滤掉存在入口的Mirror的点
                    if(this->id2spids[ie.neighbor] == ids_id
                        //  ){ // out-mirror edge
                        && v_in_mirror.find(ie.neighbor) == v_in_mirror.end()){ // out-mirror edge
                        subgraph[ie.neighbor.GetValue()].emplace_back(nbr_t(m_id,
                                                                      ie.data));
                    }
                }
            }
        }
        LOG(INFO) << "inc_subgraph_time=" << (GetCurrentTime()-inc_subgraph_time);

        // print_subgraph();
    }

    void judge_out_bound_node(const std::shared_ptr<fragment_t>& new_graph) {
        double judge_out_bound_node_time = GetCurrentTime();
        const vid_t spn_ids_num = this->supernode_ids.size();
        bool compressor_flags_cilk = true;
        if (compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
            LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                this->judge_out_bound_node_detail(j, new_graph);
            }
        }
        else{
#pragma omp parallel for num_threads(NUM_THREADS)
            for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                this->judge_out_bound_node_detail(j, new_graph);
            }
#pragma omp barrier
        }
        // 将in-mirror的master点无条件标记为边界点
        size_t not_bound = 0;
        for (vid_t i = 0; i < supernode_in_mirror.size(); i++) {
            for(auto v : supernode_in_mirror[i]) {
                if (this->supernode_out_bound[v.GetValue()] == false) {
                    not_bound++;
                }
                this->supernode_out_bound[v.GetValue()] = true;
            }
        }
        LOG(INFO) << "not_bound=" << not_bound;
        LOG(INFO) << "judge_out_bound_node_time=" << (GetCurrentTime() - judge_out_bound_node_time);

    }

    void inc_judge_out_bound_node(const std::shared_ptr<fragment_t>& new_graph) {
        double inc_judge_out_bound_node_time = GetCurrentTime();
        bool compressor_flags_cilk = true;
        vid_t update_size = this->update_cluster_ids.size();
        if (compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
            LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < update_size; j++){  // parallel compute
                this->judge_out_bound_node_detail(this->update_cluster_ids[j], new_graph);
            }
        }
        else{
#pragma omp parallel for num_threads(NUM_THREADS)
            for(vid_t j = 0; j < update_size; j++){  // parallel compute
                this->judge_out_bound_node_detail(this->update_cluster_ids[j], new_graph);
            }
#pragma omp barrier
        }
        // 将in-mirror的master点无条件标记为边界点
        size_t not_bound = 0;
        for (vid_t i = 0; i < update_size; i++) {
            for(auto v : supernode_in_mirror[this->update_cluster_ids[i]]) {
                if (this->supernode_out_bound[v.GetValue()] == false) {
                    not_bound++;
                }
                this->supernode_out_bound[v.GetValue()] = true;
            }
        }
        LOG(INFO) << "not_bound=" << not_bound;
        LOG(INFO) << "inc_judge_out_bound_node_time=" << (GetCurrentTime() - inc_judge_out_bound_node_time);

    }


    void judge_out_bound_node_detail(const vid_t ids_id, 
                                     const std::shared_ptr<fragment_t>& new_graph){
        std::vector<vertex_t> &node_set = this->supernode_ids[ids_id]; 
        std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id]; 
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(this->id2spids[e.neighbor] != ids_id 
                    && out_mirror.find(e.neighbor) == out_mirror.end()){ // 导致入口Mirror成为内部点
                    this->supernode_out_bound[v.GetValue()] = true;
                    break;
                }
            }
        }
    } 

    void print_cluster() {
        LOG(INFO) << "---------------print_cluster--------------------";
        for (vid_t i = 0; i < supernode_ids.size(); i++) {
            LOG(INFO) << "---------------------------";
            LOG(INFO) << "spids_id=" << i;
            LOG(INFO) << " P:";
            for (auto p : supernode_ids[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
            LOG(INFO) << " source:";
            for (auto p : supernode_source[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
            LOG(INFO) << " supernode_in_mirror:";
            for (auto p : supernode_in_mirror[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
            LOG(INFO) << " supernode_out_mirror:";
            for (auto p : supernode_out_mirror[i]) { 
                LOG(INFO) << v2Oid(p);
            } 
        }
        LOG(INFO) << "===============================================";

        // for (auto P : supernode_ids) {
        //     LOG(INFO) << "supernode_ids:";
        //     for (auto p : P) {
        //         LOG(INFO) << this->graph_->GetId(p);
        //     }
        // }
        // for (auto P : cluster_ids) {
        //     LOG(INFO) << "cluster_ids: ids+mirror";
        //     for (auto p : P) {
        //         LOG(INFO) << p.GetValue();
        //     }
        // }
        // for (auto P : supernode_in_mirror) {
        //     LOG(INFO) << "supernode_in_mirror:";
        //     for (auto p : P) {
        //         LOG(INFO) << this->graph_->GetId(p);
        //     }
        // }
        // for (auto P : cluster_in_mirror_ids) {
        //     LOG(INFO) << "cluster_in_mirror_ids:";
        //     for (auto p : P) {
        //         LOG(INFO) << " mid=" << p.GetValue()
        //                   << " vid=" << mirrorid2vid[p].GetValue()
        //                   << " oid=" << this->graph_->GetId(mirrorid2vid[p]);
        //     }
        // }
        // for (auto P : cluster_out_mirror_ids) {
        //     LOG(INFO) << "cluster_out_mirror_ids:";
        //     for (auto p : P) {
        //         LOG(INFO) << " mid=" << p.GetValue()
        //                   << " vid=" << mirrorid2vid[p].GetValue()
        //                   << " oid=" << this->graph_->GetId(mirrorid2vid[p]);
        //     }
        // }
    }
    
    void print_subgraph() {
        LOG(INFO) << "----------------print_subraph-------------------";
        // for(int i = 0; i < subgraph.size(); i++) {
        //     for (auto e : subgraph[i]) {
        //         LOG(INFO) << vid2Oid(i) << "->" << vid2Oid(e.neighbor.GetValue());
        //     }
        // }
        // 按cluster输出
        for(auto vs : cluster_ids) {
            LOG(INFO) << " ----";
            for (auto v : vs) {
                for (auto e : subgraph[v.GetValue()]) {
                    LOG(INFO) << " " << v2Oid(v) << "->" << v2Oid(e.neighbor);
                }
            }
        }
    }

    vid_t vid2Oid(vid_t vid) {
        if (vid < old_node_num) {
            vertex_t v(vid);
            return graph_->GetId(v);
        } else {
            return vid;
        }
    }

    vid_t v2Oid(vertex_t v) {
        if (v.GetValue() < old_node_num) {
            return graph_->GetId(v);
        } else {
            return v.GetValue();
        }
    }

    /**
     * 为worker计算时提供顶点类型
    */
    void get_nodetype(vid_t inner_node_num, std::vector<char>& node_type) {
        node_type.clear();
        node_type.resize(inner_node_num, std::numeric_limits<char>::max());
        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
          vertex_t u(i);
          if (this->Fc[u] == this->FC_default_value) {
            node_type[i] = NodeType::SingleNode; // out node
          } else if (this->Fc[u] >= 0) {
            node_type[i] = NodeType::OnlyInNode; // source node
          } else if(!this->supernode_out_bound[i]) {
            node_type[i] = NodeType::InnerNode; // inner node
          }
          if (this->supernode_out_bound[i]) {
            node_type[i] = NodeType::OnlyOutNode; // bound node
            if (this->Fc[u] >= 0) {
              node_type[i] = NodeType::BothOutInNode; // source node + bound node
            }
          }
      }
      // mirror的Master点必须要得标记为入口点,否则它不调用shourtcut.
      for(vid_t i = 0; i < this->supernode_in_mirror.size(); i++) { // can'nt parallel
        for(auto v : this->supernode_in_mirror[i]) {
          // LOG(INFO) << "----vid=" << v.GetValue();
          if(node_type[v.GetValue()] == NodeType::OnlyOutNode 
            //  || node_type[v.GetValue()] == NodeType::SingleNode 
             || node_type[v.GetValue()] == NodeType::BothOutInNode) {
                node_type[v.GetValue()] = NodeType::BothOutInNode;
          } else {
            node_type[v.GetValue()] = NodeType::OnlyInNode; // in-node or inner node
          }
        }
      }
    }

    /**
     * 为worker计算时提供顶点类型, 考虑master顶点类型
    */
    void get_nodetype_mirror(vid_t inner_node_num, std::vector<char>& node_type) {
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        if (this->Fc[u] == this->FC_default_value) {
          node_type[i] = NodeType::SingleNode; // out node
        } else if (this->Fc[u] >= 0) {
          node_type[i] = NodeType::OnlyInNode; // source node
        } else if(!this->supernode_out_bound[i]) {
          node_type[i] = NodeType::InnerNode; // inner node
        }
        if (this->supernode_out_bound[i]) {
          node_type[i] = NodeType::OnlyOutNode; // bound node
          if (this->Fc[u] >= 0) {
            node_type[i] = NodeType::BothOutInNode; // source node + bound node
          }
        }
      }
      // mirror的Master点必须要得标记为入口点,否则它不调用shourtcut.
      size_t in_cnt = 0; // just count number
      for(vid_t i = 0; i < this->supernode_in_mirror.size(); i++) { // can'nt parallel
        for(auto v : this->supernode_in_mirror[i]) {
          if (node_type[v.GetValue()] == NodeType::OnlyOutNode) {
            node_type[v.GetValue()] = NodeType::OutMaster;
          } else if (node_type[v.GetValue()] == NodeType::BothOutInNode) {
            node_type[v.GetValue()] = NodeType::BothOutInMaster;
          } else if (node_type[v.GetValue()] == NodeType::OnlyInNode) {
            in_cnt++;
          }
        }
      }
      // for (int i = 0; i < this->all_node_num; i++) {
      //   LOG(INFO) << " node typ oid=" << vid2Oid(i) 
      //             << "=" << int(node_type[i])
      //             << " this->Fc[u]=" << this->Fc[vertex_t(i)];
      // }
      LOG(INFO) << "error: in_cnt=" << in_cnt;
    }

    /**
     * 将sketch转换为CSR
     * is_e_: 表示入口点到出口点的索引, 包括master到mirror所在cluster的索引
     * ib_e_: 表示出口点和没有被压缩点之间的边
    */
    void sketch2csr_merge(std::vector<char>& node_type){
      double transfer_csr_time = GetCurrentTime();
      double init_time_1 = GetCurrentTime();
      auto inner_vertices = graph_->InnerVertices();
      vid_t inner_node_num = inner_vertices.end().GetValue() 
                             - inner_vertices.begin().GetValue();
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[i+1] += spnode.bound_delta.size();
            // atomic_add(source_e_num, spnode.bound_delta.size());
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){  // 关闭多线程
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  temp_cnt += 1;
              }
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          // atomic_add(bound_e_num, temp_cnt);
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] = temp_cnt;
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " bound_e_num=" << bound_e_num;
      LOG(INFO) << " source_e_num=" << source_e_num;
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      // build index/edge
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
                // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
                is_e_[index_s].neighbor = oe.first;
                is_e_[index_s].data = oe.second;
                index_s++;
            }
          }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        } 
        else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
        std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time);
    }

    /**
     * 将sketch转换为CSR
     * is_e_: 表示入口点到出口点的索引, 不包括master到mirror所在cluster的索引
     * ib_e_: 表示出口点和没有被压缩点之间的边
     * im_e_: 表示in-mirror相关的索引
     * om_e_: 表示out-mirror相关的索引
    */
    void sketch2csr_divide(std::vector<char>& node_type){
      double transfer_csr_time = GetCurrentTime();
      double init_time_1 = GetCurrentTime();
      auto inner_vertices = graph_->InnerVertices();
      vid_t inner_node_num = inner_vertices.end().GetValue() 
                             - inner_vertices.begin().GetValue();
      is_e_.clear();
      is_e_offset_.clear();
      im_e_.clear();
      im_e_offset_.clear();
      om_e_.clear();
      om_e_offset_.clear();
      oim_e_.clear();
      oim_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      size_t out_mirror_e_num = 0;
      size_t out_imirror_e_num = 0;
      size_t in_mirror_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> im_e_degree(inner_node_num+1, 0);
      std::vector<size_t> om_e_degree(inner_node_num+1, 0);
      std::vector<size_t> oim_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          const vid_t ids_id = this->id2spids[u];
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            // is_e_degree[i+1] += spnode.bound_delta.size();
            if (ids_id == mp.first) { // origin index
              size_t origin_size = 0;
              for (auto e : spnode.bound_delta) {
                if (this->id2spids[e.first] == ids_id) {
                    origin_size++;
                }
              }
              is_e_degree[i+1] += origin_size;
              om_e_degree[i+1] += (spnode.bound_delta.size() - origin_size);
            } else { // mirror index
            //   im_e_degree[i+1] += spnode.bound_delta.size();
              size_t in_size = 0;
              for (auto e : spnode.bound_delta) {
                if (this->id2spids[e.first] == mp.first) { // out-mirror
                    in_size++;
                }
              }
              im_e_degree[i+1] += in_size;
              oim_e_degree[i+1] += (spnode.bound_delta.size() - in_size);
            }
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){  // 关闭多线程
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  temp_cnt += 1;
              }
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          // atomic_add(bound_e_num, temp_cnt);
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] = temp_cnt;
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      // debug
      {
        //   for (vid_t i = 0; i < inner_node_num; i++) {
        //       LOG(INFO) << " oid=" << this->vid2Oid(i) << " ib=" << ib_e_degree[i+1];
        //       LOG(INFO) << " oid=" << this->vid2Oid(i) << " is=" << is_e_degree[i+1];
        //       LOG(INFO) << " oid=" << this->vid2Oid(i) << " im=" << im_e_degree[i+1];
        //       LOG(INFO) << " oid=" << this->vid2Oid(i) << " om=" << om_e_degree[i+1];
        //       LOG(INFO) << "----------------";
        //   }
      }

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
        im_e_degree[i] += im_e_degree[i-1];
        om_e_degree[i] += om_e_degree[i-1];
        oim_e_degree[i] += oim_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      in_mirror_e_num = im_e_degree[inner_node_num];
      out_mirror_e_num = om_e_degree[inner_node_num];
      out_imirror_e_num = oim_e_degree[inner_node_num];
      LOG(INFO) << " bound_e_num=" << bound_e_num;
      LOG(INFO) << " source_e_num=" << source_e_num;
      LOG(INFO) << " in_mirror_e_num=" << in_mirror_e_num;
      LOG(INFO) << " out_mirror_e_num=" << out_mirror_e_num;
      LOG(INFO) << " out_imirror_e_num=" << out_imirror_e_num;
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      im_e_.resize(in_mirror_e_num);
      om_e_.resize(out_mirror_e_num);
      oim_e_.resize(out_imirror_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      im_e_offset_.resize(inner_node_num+1);
      om_e_offset_.resize(inner_node_num+1);
      oim_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      // build index/edge
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        vid_t index_im = im_e_degree[i];
        im_e_offset_[i] = &im_e_[index_im];
        vid_t index_om = om_e_degree[i];
        om_e_offset_[i] = &om_e_[index_om];
        vid_t index_oim = oim_e_degree[i];
        oim_e_offset_[i] = &oim_e_[index_oim];
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
          const vid_t ids_id = this->id2spids[u];
          for(auto mp : this->shortcuts[i]) {
            // vid_t sp_id = mp.second;
            // supernode_t &spnode = this->supernodes[sp_id];
            // for(auto& oe : spnode.bound_delta){
            //     // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            //     is_e_[index_s].neighbor = oe.first;
            //     is_e_[index_s].data = oe.second;
            //     index_s++;
            // }
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            // is_e_degree[i+1] += spnode.bound_delta.size();
            if (ids_id == mp.first) { // origin index
              for (auto oe : spnode.bound_delta) {
                if (this->id2spids[oe.first] == ids_id) {
                    is_e_[index_s].neighbor = oe.first;
                    is_e_[index_s].data = oe.second;
                    index_s++;
                } else {
                    om_e_[index_om].neighbor = oe.first;
                    om_e_[index_om].data = oe.second;
                    index_om++;
                }
              }
            } else { // mirror index
              for(auto& oe : spnode.bound_delta){
                // im_e_[index_im].neighbor = oe.first;
                // im_e_[index_im].data = oe.second;
                // index_im++;
                if (this->id2spids[oe.first] == mp.first) { // out-mirror
                    im_e_[index_im].neighbor = oe.first;
                    im_e_[index_im].data = oe.second;
                    index_im++;
                } else {
                    oim_e_[index_oim].neighbor = oe.first;
                    oim_e_[index_oim].data = oe.second;
                    index_oim++;
                }
              }
            }
          }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        } 
        else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      im_e_offset_[inner_node_num] = &im_e_[in_mirror_e_num-1] + 1;
      om_e_offset_[inner_node_num] = &om_e_[out_mirror_e_num-1] + 1;
      oim_e_offset_[inner_node_num] = &oim_e_[out_imirror_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
        std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time);
    }

    /**
     * 将sketch转换为CSR
     * is_e_: 表示入口点到出口点的索引, 包括master到mirror所在cluster的索引
     * ib_e_: 表示出口点和没有被压缩点之间的边
     * sync_e_: mirror-master之间的同步边
     * 注意: 参与计算的顶点将需要包括Mirror点.
    */
    void sketch2csr_mirror(std::vector<char>& node_type){
      double transfer_csr_time = GetCurrentTime();
      double init_time_1 = GetCurrentTime();
      auto inner_vertices = graph_->InnerVertices();
      // vid_t inner_node_num = inner_vertices.end().GetValue() 
      //                        - inner_vertices.begin().GetValue();
      vid_t inner_node_num = this->all_node_num;
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      sync_e_.clear();
      sync_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      size_t sync_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      std::vector<size_t> sync_e_degree(inner_node_num+1, 0);
      LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1);


      {
        double sync_e_time = GetCurrentTime();
        vid_t cluster_size = this->supernode_in_mirror.size();
        std::vector<std::vector<vertex_t> > syncE;
        syncE.resize(inner_node_num);
        for(vid_t i = 0; i < cluster_size; i++) { // can'nt parallel
          for(auto v : this->cluster_in_mirror_ids[i]) {
            syncE[this->mirrorid2vid[v].GetValue()].emplace_back(v);
            sync_e_degree[this->mirrorid2vid[v].GetValue()+1]++; // master -> in-mirror
          }
          for(auto v : this->cluster_out_mirror_ids[i]) {
            syncE[v.GetValue()].emplace_back(this->mirrorid2vid[v]);
            sync_e_degree[v.GetValue()+1]++;  // out-mirror -> master
          }
        }
        for(vid_t i = 1; i <= inner_node_num; i++) {
          sync_e_degree[i] += sync_e_degree[i-1];
        }
        sync_e_num = sync_e_degree[inner_node_num];
        sync_e_.resize(sync_e_num);
        sync_e_offset_.resize(inner_node_num+1);
        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
          vid_t index = sync_e_degree[i];
          sync_e_offset_[i] = &sync_e_[index];
          for (auto v : syncE[i]) {
            sync_e_[index].neighbor = v;
            index++;
          }
        }
        sync_e_offset_[inner_node_num] = &sync_e_[sync_e_num-1] + 1;
        LOG(INFO) << " sync_e_num=" << sync_e_num;
        LOG(INFO) << "sync_e_time=" << (GetCurrentTime()-sync_e_time);
      }

      double csr_time_1 = GetCurrentTime();
      // in-mirror
      parallel_for(vid_t i = this->old_node_num; i < this->all_node_num; i++) {
        vertex_t u(i);
        vid_t sp_id = Fc_map[u];
        if (sp_id != ID_default_value) {
          // LOG(INFO) << " u.oid=" << this->v2Oid(u) << " sp_id=" << sp_id;
          supernode_t &spnode = this->supernodes[sp_id];
          is_e_degree[i+1] += spnode.bound_delta.size();
        }
      }
      // LOG(INFO) << "------------";
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        const char type = node_type[i];
        // LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(i)
        //           << " type=" << int(type);
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster){
          // for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = Fc_map[u];
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[i+1] += spnode.bound_delta.size();
            // atomic_add(source_e_num, spnode.bound_delta.size());
          // }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster || type == NodeType::OutMaster){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){  // 关闭多线程
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  temp_cnt += 1;
              }
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          // atomic_add(bound_e_num, temp_cnt);
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] = temp_cnt;
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " bound_e_num=" << bound_e_num;
      LOG(INFO) << " source_e_num=" << source_e_num;
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      // build index/edge
      double csr_time_2 = GetCurrentTime();
      // in-mirror
      parallel_for(vid_t i = this->old_node_num; i < this->all_node_num; i++) {
        vertex_t u(i);
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];  // must init
        vid_t sp_id = Fc_map[u];
        if (sp_id != ID_default_value) {
          supernode_t &spnode = this->supernodes[sp_id];
          for(auto& oe : spnode.bound_delta){
            // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            is_e_[index_s].neighbor = oe.first;
            is_e_[index_s].data = oe.second;
            index_s++;
          }
        }
      }
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[i] = &is_e_[index_s];
        const char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster){
          // for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = Fc_map[u];
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
                // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
                is_e_[index_s].neighbor = oe.first;
                is_e_[index_s].data = oe.second;
                index_s++;
            }
          // }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[i] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode
           || type == NodeType::BothOutInMaster || type == NodeType::OutMaster){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = 
                                            this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        } 
        else if (NodeType::SingleNode == type) { // edge
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
        std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time);

      // debug 
      // {
      //   LOG(INFO) << "----------------------------------------";
      //   for(vid_t i = 0; i < this->all_node_num; i++) {
      //     LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(i);
      //     adj_list_t adj = adj_list_t(sync_e_offset_[i], sync_e_offset_[i+1]);
      //     for(auto e : adj) {
      //       LOG(INFO) << " sync: " << this->vid2Oid(i) << "->" 
      //                 << this->v2Oid(e.neighbor);
      //     }
      //     adj_list_index_t is_adj = adj_list_index_t(is_e_offset_[i], is_e_offset_[i+1]);
      //     for(auto e : is_adj) {
      //       LOG(INFO) << " is_adj: " << this->vid2Oid(i) << "->" 
      //                 << this->v2Oid(e.neighbor);
      //     }
      //     adj_list_t ib_adj = adj_list_t(ib_e_offset_[i], ib_e_offset_[i+1]);
      //     for(auto e : ib_adj) {
      //       LOG(INFO) << " ib_adj: " << this->vid2Oid(i) << "->" 
      //                 << this->v2Oid(e.neighbor);
      //     }
      //   }
      //   LOG(INFO) << "========================================";
      // }
    }

    /**
     * 用于sum_sync_traversal_worker.h
     * 将sketch转为CSR, 同时对顶点重新编号,编号按照同一类在一个区间,具体规则为:
     * 0---外部点---出口点---入口点---入口+出口点---内部点---最大顶点id.
     * is_e_: 表示入口点到出口点的索引, 包括master到mirror所在cluster的索引
     * ib_e_: 表示出口点和没有被压缩点之间的边
     * 注意: 这里之所以传递is_e_这几个csr数据,是因为和tra-woker里面类型冲突了,
     * 后面在详细看看?
    */
    void sketch2csr_renumber(vid_t inner_node_num,
                            std::vector<char>& node_type,
                            Array<vid_t, Allocator<vid_t>>& oldId2newId,
                            Array<vid_t, Allocator<vid_t>>& newId2oldId,
                            Array<vid_t, Allocator<vid_t>>& oldGid2newGid,
                            Array<vid_t, Allocator<vid_t>>& newGid2oldGid,
                            std::vector<vid_t>& node_range,
                            std::vector<std::vector<vertex_t>>& all_nodes,
                            Array<nbr_index_t, Allocator<nbr_index_t>>& is_e_,
                            Array<nbr_index_t*, Allocator<nbr_index_t*>>& is_e_offset_,
                            Array<nbr_t, Allocator<nbr_t>>& ib_e_,
                            Array<nbr_t*, Allocator<nbr_t*>>& ib_e_offset_
                            ) {
      LOG(INFO) << "inner_node_num=" << inner_node_num;
      double node_type_time = GetCurrentTime();
      all_nodes.clear();
      all_nodes.resize(5);
      for(vid_t i = 0; i < inner_node_num; i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      LOG(INFO) << " node_type_time=" << (GetCurrentTime()-node_type_time);
      
      /* renumber internal vertices */
      oldId2newId.clear();
      oldId2newId.resize(inner_node_num);
      newId2oldId.clear();
      newId2oldId.resize(inner_node_num);
      oldGid2newGid.clear();
      oldGid2newGid.resize(inner_node_num); // 分布式情况下，可能会越界，换成unordered_map？？？
      newGid2oldGid.clear();
      newGid2oldGid.resize(inner_node_num);
      node_range.clear();
      node_range.resize(6);
      vid_t index_id = 0;
      // for (int i = 0; i < 5; i++) {
      //   index_id[i+1] = index_id[i] + all_nodes[i].size();
      // }
      for (vid_t i = 0; i < 5; i++) {
        const std::vector<vertex_t>& nodes = all_nodes[i];
        size_t size = nodes.size();
        node_range[i] = index_id;
        parallel_for (vid_t j = 0; j < size; j++) {
          oldId2newId[nodes[j].GetValue()] = index_id + j;
          newId2oldId[index_id + j] = nodes[j].GetValue();
          // oldId2newId[nodes[j].GetValue()] = nodes[j].GetValue(); // 调试,全部设置为原来的值
          // newId2oldId[nodes[j].GetValue()] = nodes[j].GetValue();
          // 需要加个判断，只转化本地gid
          vid_t old_gid = this->graph_->Vertex2Gid(nodes[j]);
          vid_t new_gid = this->graph_->Vertex2Gid(vertex_t(index_id + j));
          oldGid2newGid[old_gid] = new_gid;
          newGid2oldGid[new_gid] = old_gid;
          // oldGid2newGid[old_gid] = old_gid; // 调试,全部设置为原来的值
          // newGid2oldGid[old_gid] = old_gid;
        }
        index_id += size;
      }
      node_range[5] = index_id;

      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */
      double  init_time_1 = GetCurrentTime();
      // spnode_datas.Init(inner_vertices, 0);
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << " init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[oldId2newId[i]+1] += spnode.bound_delta.size(); // Note: Accumulation is used here.
            // LOG(INFO) << " +size=" << spnode.bound_delta.size()
            //           << " is_e_degree=" << is_e_degree[oldId2newId[i]+1];
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge // 应该是 else if
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            }
          }
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] += temp_cnt;
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] = temp_cnt;
        }
      }
      LOG(INFO) << " csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " bound_e_num=" << bound_e_num;
      LOG(INFO) << " source_e_num=" << source_e_num;
      LOG(INFO) << " index_time=" << (GetCurrentTime()-index_time); //0.226317

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << " init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      /* build index/edge */
      double csr_time_2 = GetCurrentTime();
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        /* source node */
        vid_t new_id = oldId2newId[i];
        vid_t index_s = is_e_degree[new_id];
        is_e_offset_[new_id] = &is_e_[index_s];
        char type = node_type[i];
        // LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(new_id) << " type=" << int(type);
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            // LOG(INFO) << " mp.second=" << mp.second;
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
              // LOG(INFO) << " -sp.oid=" << this->v2Oid(spnode.id) << "->"
              //           << this->v2Oid(oe.first) << " data=" << oe.second;
              if (oe.first.GetValue() < inner_node_num) {
                is_e_[index_s].neighbor = oldId2newId[oe.first.GetValue()];
              } else {
                is_e_[index_s].neighbor = oe.first;
              }
              // The dependent parent id also donot needs to be updated, 
              // because it is gid.
              is_e_[index_s].data = oe.second;
              if (oe.second.parent_gid < inner_node_num) {
                is_e_[index_s].data.parent_gid = oldId2newId[oe.second.parent_gid];
              }
              // LOG(INFO) << " +sp.oid=" << this->v2Oid(spnode.id) << "->"
                        // << this->v2Oid(is_e_[index_s].neighbor) << " data=" << is_e_[index_s].data;
              // LOG(INFO) << " index_s=" << index_s << " oe.size=" << spnode.bound_delta.size();
              index_s++;
            }
          }
        }
        /* inner_bound node */
        // vid_t index_b = ib_e_degree[i];
        vid_t index_b = ib_e_degree[new_id];
        ib_e_offset_[new_id] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()
                && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  auto nbr = ib_e_[index_b].neighbor;
                  if (nbr.GetValue() < inner_node_num) {
                    ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
                  }
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  auto nbr = ib_e_[index_b].neighbor;
                  if (nbr.GetValue() < inner_node_num) {
                    ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
                  }
                  index_b++;
              }
            }
          }
        }
        if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            auto nbr = ib_e_[index_b].neighbor;
            if (nbr.GetValue() < inner_node_num) {
              ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
            }
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      {
        // ib_e_offset_ should have been sorted.
        // for (vid_t i = 0; i < inner_node_num; ++i) {
        //   std::sort(ib_e_offset_[i], ib_e_offset_[i + 1],
        //           [](const nbr_t& lhs, const nbr_t& rhs) {
        //             return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
        //           });
        // }
        parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }
      LOG(INFO) << " csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      // debug 
      // {
      //   LOG(INFO) << "----------------------------------------";
      //   for(vid_t i = 0; i < this->old_node_num; i++) {
      //     LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(newId2oldId[i]);
      //     adj_list_index_t is_adj = adj_list_index_t(is_e_offset_[i], is_e_offset_[i+1]);
      //     for(auto e : is_adj) {
      //       LOG(INFO) << " is_adj: " << this->vid2Oid(newId2oldId[i]) << "->" 
      //                 << this->vid2Oid(newId2oldId[e.neighbor.GetValue()])
      //                 << " weight=" << e.data;
      //     }
      //     LOG(INFO) << " ----";
      //     adj_list_t ib_adj = adj_list_t(ib_e_offset_[i], ib_e_offset_[i+1]);
      //     for(auto e : ib_adj) {
      //       LOG(INFO) << " ib_adj: " << this->vid2Oid(newId2oldId[i]) << "->" 
      //                 << this->vid2Oid(newId2oldId[e.neighbor.GetValue()])
      //                 << " weight=" << e.data;
      //     }
      //   }
      //   LOG(INFO) << "========================================";
      // }
      
      // just count mirror for expr and count skeleton
      if (FLAGS_count_skeleton) {
        LOG(INFO) << "\nopen cout skeleton:";
        size_t bound_node_out_edge_num = 0; // type=1,3的出边, bound_edge
        size_t skeleton_edge_num = 0; // shortcut + all out_edge
        size_t skeleton_node_num = 0; // all_node - type=4内部点数
        size_t mirror_node_num = 0;   // mirror点数
        size_t local_edges_num = graph_->GetEdgeNum() / 2;

        for (vid_t i = node_range[1]; i < node_range[2]; i++) {
          vertex_t u(i);
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          bound_node_out_edge_num += oes.Size();
        }
        for (vid_t i = node_range[3]; i < node_range[4]; i++) {
          vertex_t u(i);
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          bound_node_out_edge_num += oes.Size();
        }
        LOG(INFO) << " for mirror-master expr:";
        LOG(INFO) << "#bound_node_out_edge_num: " << bound_node_out_edge_num;

        skeleton_edge_num += source_e_num;
        skeleton_edge_num += bound_e_num;
        LOG(INFO) << "#skeleton_edge_num: " << skeleton_edge_num;
        LOG(INFO) << "  source_e_num: " << source_e_num;
        LOG(INFO) << "  bound_e_num: " << bound_e_num;

        skeleton_node_num = this->old_node_num - all_nodes[4].size();
        // for (auto v : all_nodes[4]) {
        //   LOG(INFO) << " v=" << this->v2Oid(v);
        // }
        LOG(INFO) << "#skeleton_node_num: " << skeleton_node_num;
        LOG(INFO) << "#local_all_edges_num=" << local_edges_num;
        LOG(INFO) << "#new_cmp_rate=" 
                  << (local_edges_num-skeleton_edge_num)*1.0/local_edges_num;
      }
    }

    /**
     * 用于sum_sync_traversal_worker.h
     * 将sketch转为CSR, 注意csr中不包含mirro点
    */
    void sketch2csr(vid_t inner_node_num,
                            std::vector<char>& node_type,
                            std::vector<std::vector<vertex_t>>& all_nodes,
                            Array<nbr_index_t, Allocator<nbr_index_t>>& is_e_,
                            Array<nbr_index_t*, Allocator<nbr_index_t*>>& is_e_offset_,
                            Array<nbr_t, Allocator<nbr_t>>& ib_e_,
                            Array<nbr_t*, Allocator<nbr_t*>>& ib_e_offset_
                            ) {
      LOG(INFO) << "inner_node_num=" << inner_node_num;
      double node_type_time = GetCurrentTime();
      all_nodes.clear();
      all_nodes.resize(5);
      for(vid_t i = 0; i < inner_node_num; i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      LOG(INFO) << " node_type_time=" << (GetCurrentTime()-node_type_time);
      
      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */
      double  init_time_1 = GetCurrentTime();
      // spnode_datas.Init(inner_vertices, 0);
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << " init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            is_e_degree[i+1] += spnode.bound_delta.size(); // Note: Accumulation is used here.
          }
        }
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge // 应该是 else if
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id 
                 && out_mirror.find(e.neighbor) == out_mirror.end()
                 && in_mirror.find(u) == in_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            } else {
              if(ids_id != out_ids_id 
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  // bound_e_num += 1;
                  temp_cnt += 1;
              }
            }
          }
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[i+1] += temp_cnt;
        } else if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[i+1] = temp_cnt;
        }
      }
      LOG(INFO) << " csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " bound_e_num=" << bound_e_num;
      LOG(INFO) << " source_e_num=" << source_e_num;
      LOG(INFO) << " index_time=" << (GetCurrentTime()-index_time); //0.226317

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << " init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      /* build index/edge */
      double csr_time_2 = GetCurrentTime();
      parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        vertex_t u(i);
        /* source node */
        vid_t new_id = i;
        vid_t index_s = is_e_degree[new_id];
        is_e_offset_[new_id] = &is_e_[index_s];
        char type = node_type[i];
        // LOG(INFO) << "i=" << i << " oid=" << this->vid2Oid(new_id) << " type=" << int(type);
        if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){ // index
          for(auto mp : this->shortcuts[i]) {
            // LOG(INFO) << " mp.second=" << mp.second;
            vid_t sp_id = mp.second;
            supernode_t &spnode = this->supernodes[sp_id];
            for(auto& oe : spnode.bound_delta){
              is_e_[index_s].neighbor = oe.first;
              is_e_[index_s].data = oe.second;
              index_s++;
            }
          }
        }
        /* inner_bound node */
        // vid_t index_b = ib_e_degree[i];
        vid_t index_b = ib_e_degree[new_id];
        ib_e_offset_[new_id] = &ib_e_[index_b];
        if(type == NodeType::OnlyOutNode || type == NodeType::BothOutInNode){ // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = this->id2spids[u];
          std::unordered_set<vertex_t> &out_mirror = this->supernode_out_mirror[ids_id];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            vid_t out_ids_id = this->id2spids[e.neighbor];
            if (out_ids_id != this->ID_default_value) {
              std::unordered_set<vertex_t> &in_mirror = 
                                        this->supernode_in_mirror[out_ids_id];
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()
                && in_mirror.find(u) == in_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            } else {
              if(ids_id != out_ids_id
                && out_mirror.find(e.neighbor) == out_mirror.end()){
                  ib_e_[index_b] = e;
                  index_b++;
              }
            }
          }
        }
        if (NodeType::SingleNode == type) { // edge
          auto oes = this->graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      {
        // ib_e_offset_ should have been sorted.
        // for (vid_t i = 0; i < inner_node_num; ++i) {
        //   std::sort(ib_e_offset_[i], ib_e_offset_[i + 1],
        //           [](const nbr_t& lhs, const nbr_t& rhs) {
        //             return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
        //           });
        // }
        parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }
      LOG(INFO) << " csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      // just count mirror for expr and count skeleton
      if (FLAGS_count_skeleton) {
        LOG(INFO) << "==================COUNT SKELETON========================";
        size_t bound_node_out_edge_num = 0; // type=1,3的出边, bound_edge
        size_t skeleton_edge_num = 0; // shortcut + all out_edge
        size_t skeleton_node_num = 0; // all_node - type=4内部点数
        size_t mirror_node_num = 0;   // mirror点数
        size_t local_edges_num = graph_->GetEdgeNum() / 2;

        for (vid_t i = 0; i < inner_node_num; i++) {
          if (node_type[i] == NodeType::OnlyOutNode 
              || node_type[i] == NodeType::BothOutInNode) {
            vertex_t u(i);
            adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
            bound_node_out_edge_num += oes.Size();
          }
        }
        LOG(INFO) << " for mirror-master expr:";
        LOG(INFO) << "#bound_node_out_edge_num: " << bound_node_out_edge_num;

        skeleton_edge_num += source_e_num;
        skeleton_edge_num += bound_e_num;
        LOG(INFO) << "#skeleton_edge_num: " << skeleton_edge_num;
        LOG(INFO) << "  source_e_num: " << source_e_num;
        LOG(INFO) << "  bound_e_num: " << bound_e_num;

        skeleton_node_num = this->old_node_num - all_nodes[4].size();
        // for (auto v : all_nodes[4]) {
        //   LOG(INFO) << " v=" << this->v2Oid(v);
        // }
        LOG(INFO) << "#skeleton_node_num: " << skeleton_node_num;
        LOG(INFO) << "#local_all_edges_num=" << local_edges_num;
        LOG(INFO) << "#new_cmp_rate=" 
                  << (local_edges_num-skeleton_edge_num)*1.0/local_edges_num;
        LOG(INFO) << "=========================END============================";
      }
    }

    /**
     * 
     * 对于增加的边或者删除的边：u->v
     * 需要更新u/v所在的cluster以及它们的in-mirror点所在cluster。
     * Out-Mirror点只收集消息并不参与计算(out-mirror只有入边)，
     * 故其所在的cluster不需要被动更新。
    */
    void inc_trav_compress_mirror(
            std::vector<std::pair<vid_t, vid_t>>& deleted_edges, 
            std::vector<std::pair<vid_t, vid_t>>& added_edges,
            const std::shared_ptr<fragment_t>& new_graph){
        LOG(INFO) << "inc_trav_compress_mirror...";
        LOG(INFO) << " old spnode_num=" << this->supernodes_num;
        size_t old_supernodes_num = this->supernodes_num;

        // this->print_cluster();
        // this->print("inc之前");

        /* 确定受影响的cluster: master以及其in-mirror点所在的
            cluster都得更新。
         */
        fid_t fid = this->graph_->fid();
        auto vm_ptr = this->graph_->vm_ptr();
        std::unordered_set<vid_t> temp_update_cluster_ids;

        for(auto& pair : deleted_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(u_fid == fid && this->Fc[u] != this->FC_default_value){
                // LOG(INFO) << " u_id=" << this->v2Oid(u); 
                temp_update_cluster_ids.insert(this->id2spids[u]);
                for (auto spid : this->vid2in_mirror_cluster_ids[u.GetValue()]) {
                    // LOG(INFO) << " u_id-mirror spid=" << spid; 
                    temp_update_cluster_ids.insert(spid);
                }
                // reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                temp_update_cluster_ids.insert(this->id2spids[v]);
                // for (auto spid : this->vid2in_mirror_cluster_ids[v.GetValue()]) { // 对于u->v, v的in-mirror没必要更新！！！
                //     temp_update_cluster_ids.insert(spid);
                // }
                // reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
        }
        for(auto& pair : added_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(u_fid == fid && this->Fc[u] != this->FC_default_value){
                temp_update_cluster_ids.insert(this->id2spids[u]);
                for (auto spid : this->vid2in_mirror_cluster_ids[u.GetValue()]) {
                    temp_update_cluster_ids.insert(spid);
                }
                // reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                temp_update_cluster_ids.insert(this->id2spids[v]);
                // for (auto spid : this->vid2in_mirror_cluster_ids[v.GetValue()]) {
                //     temp_update_cluster_ids.insert(spid);
                // }
                // reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
        }

        // this->update_cluster_ids.clear();
        this->update_cluster_ids.assign(temp_update_cluster_ids.begin(), 
                                       temp_update_cluster_ids.end());

        // for (auto u : this->update_cluster_ids) {
        //     LOG(INFO) << "update: spids=" << u;
        // }
        // LOG(INFO) << "-------------------test---------------";

        /* 确定需要更新的索引以及新建的索引: in-mirror点还是原来的直接增量(不需要考虑增删)，
            对于普通点则需要重新统计入口点，并进行增删。
         */
        for (auto ids_id : this->update_cluster_ids) {
            supernode_t& spnode = this->supernodes[ids_id];
            std::vector<vertex_t> &node_set = this->supernode_ids[ids_id];
            std::vector<vertex_t> &old_S = this->supernode_source[ids_id];
            std::unordered_set<vertex_t> &in_mirror = 
                                            this->supernode_in_mirror[ids_id];
            // 统计新的入口点
            std::set<vertex_t> S;
            for(auto v : node_set){ // 遍历原来的点集
                const auto& oes = new_graph->GetIncomingAdjList(v); // get new adj
                for(auto& e : oes){
                    if(this->id2spids[e.neighbor] != ids_id 
                        && in_mirror.find(e.neighbor) == in_mirror.end()){ // 包含Mirror
                        S.insert(v);
                        break;
                    }
                }
            }
            std::vector<vid_t> delete_spid; // 需要删除的入口点, 其它的复用
            for (auto s : old_S) {
                CHECK(this->Fc[s] >= 0);
                if (S.find(s) == S.end()) {
                    // 注意: 还需要清理shortcut中的index.
                    // 从map中删除
                    vid_t spid = this->Fc_map[s];
                    this->supernodes[spid].clear(); // 清空旧的shortcut信息
                    delete_spid.emplace_back(spid);
                    this->Fc[s] = -(ids_id+1);
                    this->Fc_map[s] = this->ID_default_value;
                    // this->shortcuts[s.GetValue()].erase(spid); // 线程不安全
                    this->shortcuts[s.GetValue()].erase(ids_id); // 线程不安全
                }
            }
            int delete_id = delete_spid.size() - 1;
            for(auto src : S){
                if (this->Fc_map[src] != this->ID_default_value) {
                    // LOG(INFO) << " inc: spid=" << this->Fc_map[src]
                    //           << " source=" << this->v2Oid(src);
                    continue; // 原来就存在，可以复用，增量更新即可
                }
                vid_t supernode_id;
                if (delete_id >= 0) {
                    supernode_id = delete_spid[delete_id];
                    delete_id--;
                } else {
                    supernode_id = __sync_fetch_and_add(&this->supernodes_num, 1);
                }
                this->Fc[src] = ids_id;
                // LOG(INFO) << " build a new sp=" << supernode_id
                //           << " source=" << this->v2Oid(src);
                /* build a supernode */
                this->Fc_map[src] = supernode_id;
                {
                    std::unique_lock<std::mutex> lk(this->supernode_ids_mux_);
                    this->shortcuts[src.GetValue()][ids_id] = supernode_id; // 线程不安全
                }
                this->supernodes[supernode_id].status = false;
                this->supernodes[supernode_id].id = src;
                this->supernodes[supernode_id].ids = ids_id; // root_id
            }
            // 剩余的需要删除
            for (int i = 0; i <= delete_id; i++) {
              this->delete_supernode(delete_spid[i]);
            }
            old_S.clear();
            old_S.insert(old_S.begin(), S.begin(), S.end());
        }
        // this->print_cluster();
        // this->print("inc之后");
        LOG(INFO) << " new spnode_num=" << this->supernodes_num;
        size_t old_ids_num = this->cluster_ids.size();
        LOG(INFO) << " all ids_num=" << old_ids_num;
        LOG(INFO) << " update ids_num=" << this->update_cluster_ids.size();
        LOG(INFO) << " update rate(update_ids/all_ids)=" 
                  << (this->update_cluster_ids.size() * 1.0 / old_ids_num);
        LOG(INFO) << "finish inc_trav_compress_mirror.";
    }

    /**
     * 针对删边/加边获取需要重新计算的索引以及需要更新的cluster
     * 针对每条边具体分析可能导致的影响:
     *  针对: -u->v:
     *    if u是内部点,则更新cluster_u的所有入口点的索引,包括in-mirror.
     *    if u是外部点,且v是入口点,看删除u是否会使得v成为非入口点, 看v的入邻居中是否有外部
     *      点且它在cluster_v里面没有in-mirror点,如果没有,则需要删除以v为入口点的索引.
     *  针对: +u->v:
     *    if u是内部点,则更新cluster_u的所有入口点的索引,包括in-mirror.
     *    if v是内部点且不是入口点, u不在v的cluster里面, 则如果u不在v里面存在in-mirror点,
     *      则需要增加一个以v为入口点的索引.
     * update_cluster_ids: 存放收到影响需要做局部计算的cluster id.
     * update_source_id: 存放需要更新索引的supernode id, 通过supernode.status来标记
     *  是增量计算还是重新计算, 新建的都标记为false. 注意: update_source_id里面存放的
     *  source可能无效, 因为在放进去之后,有可能后面被删除了边,不成为source点.
    */
    void inc_compress_mirror(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, 
            std::vector<std::pair<vid_t, vid_t>>& added_edges,
            const std::shared_ptr<fragment_t>& new_graph){
      LOG(INFO) << "inc_compress_mirror...";
      LOG(INFO) << "  old spnode_num=" << this->supernodes_num;

      // this->print_cluster();
      // this->print("inc_compress before...");
      
      fid_t fid = graph_->fid();
      auto vm_ptr = graph_->vm_ptr();
      update_cluster_ids.clear();
      update_source_id.clear();
      std::unordered_set<vid_t> temp_update_cluster_ids;
      std::unordered_set<vid_t> temp_update_source_id;
      size_t old_supernodes_num = this->supernodes_num;
      vid_t add_num = 0; // just count num
      vid_t del_num = 0; // just count num

      double del_time_1 = 0;
      double del_time_2 = 0;

      std::vector<vid_t> delete_spid; // 需要删除的入口点, 延迟删除

      LOG(INFO) << "  deal deleted_edges...";
      double del_time = GetCurrentTime();
      for(auto& pair : deleted_edges) {
        auto u_gid = pair.first;
        auto v_gid = pair.second;
        fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
              v_fid = vm_ptr->GetFidFromGid(v_gid);
        // u -> v
        // LOG(INFO) << u_gid << "->" << v_gid;
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        vertex_t v;
        CHECK(graph_->Gid2Vertex(v_gid, v));
        if(u_fid == fid && Fc[u] != FC_default_value){
          vid_t src_id = this->id2spids[u];
          temp_update_cluster_ids.insert(src_id);
          // all source in src_id, include in-mirror
          for(auto source : this->supernode_source[src_id]){
            temp_update_source_id.insert(source.GetValue());
          }
          // cluster u's in-mirror
          for(auto mid : this->cluster_in_mirror_ids[src_id]){
            temp_update_source_id.insert(mid.GetValue());
          }
          // node u's in-mirror
          for (auto mid : this->vid2in_mirror_mids[u.GetValue()]) {
            temp_update_source_id.insert(mid);
            temp_update_cluster_ids.insert(this->id2spids[vertex_t(mid)]);
          }
        }
        // 仅仅当v是入口点时需要考虑是否删除
        if(v_fid == fid && Fc[v] != FC_default_value && Fc[v] >= 0){ // FC_default_value > 0
          // vid_t del_id = Fc_map[Fc[v][0]];
          vid_t ids_id = this->id2spids[v];
          temp_update_cluster_ids.insert(ids_id); // 为了减少校正后迭代次数,添加的(感觉正常来说不需要添加.)
          // if(ids_id != this->id2spids[u] && src.size() > 1){
          if(ids_id != this->id2spids[u]){
            CHECK(Fc[v] >= 0);
            const auto& ies = new_graph->GetIncomingAdjList(v); // 用新图
            bool hava_out_inadj = false;
            del_time_1 -= GetCurrentTime();
            std::unordered_set<vertex_t> &in_mirror = 
                                            this->supernode_in_mirror[ids_id];
            for (auto& e : ies) {
              auto& nb = e.neighbor;
              if(nb != u && ids_id != this->id2spids[nb]
                  && in_mirror.find(e.neighbor) == in_mirror.end()){
                hava_out_inadj = true;
                break;
              }
            }
            /*-----parallel-----*/
            // auto out_degree = ies.Size();
            // auto it = ies.begin();
            // // 下面的多线程可能没用
            // granular_for(j, 0, out_degree, (out_degree > 1024), {
            //   auto& e = *(it + j);
            //   auto& nb = e.neighbor;
            //   if(!hava_out_inadj && nb != u && ids_id != this->id2spids[nb]
            //       && in_mirror.find(e.neighbor) == in_mirror.end()){
            //     hava_out_inadj = true;
            //     // break; // 利用have-out-inadj达到目的
            //   }
            // })
            del_time_1 += GetCurrentTime();
            del_time_2 -= GetCurrentTime();
            if(hava_out_inadj == false){
              {
                // 需要从source集合中删除掉
                CHECK(remove_array(supernode_source[ids_id], v));
              }
              vid_t del_spid = this->Fc_map[v];
              this->supernodes[del_spid].clear(); // 清空旧的shortcut信息
              delete_spid.emplace_back(del_spid);
              this->Fc[v] = -(ids_id+1);
              this->Fc_map[v] = this->ID_default_value;
              // this->shortcuts[v.GetValue()].erase(del_spid); // 线程不安全, 不是删除value
              this->shortcuts[v.GetValue()].erase(ids_id); // 线程不安全, 删除Key=ids_id
              del_num++;
            }
            del_time_2 += GetCurrentTime();
          }
        }
      }
      LOG(INFO) << "    del_time_1-1=" << del_time_1;
      LOG(INFO) << "    del_time_2=" << del_time_2;
      LOG(INFO) << "  del_time=" << (GetCurrentTime() - del_time);

      LOG(INFO) << "deal added_edges...";
      int del_index = delete_spid.size() - 1; // 注意必须用有符号整数
      double add_time = GetCurrentTime();
      for(auto& pair : added_edges){
        auto u_gid = pair.first;
        auto v_gid = pair.second;
        fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
              v_fid = vm_ptr->GetFidFromGid(v_gid);
        // u -> v
        // LOG(INFO) << u_gid << "->" << v_gid;
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        vertex_t v;
        CHECK(graph_->Gid2Vertex(v_gid, v));
        if(u_fid == fid && Fc[u] != FC_default_value){
          vid_t src_id = this->id2spids[u];
          temp_update_cluster_ids.insert(src_id);
          // all source in src_id, include in-mirror
          for(auto source : this->supernode_source[src_id]){
            temp_update_source_id.insert(source.GetValue());
          }
          // cluster u's in-mirror
          for(auto mid : this->cluster_in_mirror_ids[src_id]){
            temp_update_source_id.insert(mid.GetValue());
          }
          // node u's in-mirror
          for (auto mid : this->vid2in_mirror_mids[u.GetValue()]) {
            temp_update_source_id.insert(mid);
            temp_update_cluster_ids.insert(this->id2spids[vertex_t(mid)]);
          }
        }
        if(v_fid == fid && Fc[v] != FC_default_value){
          vid_t ids_id = this->id2spids[v];
          temp_update_cluster_ids.insert(ids_id); // 为了减少校正后迭代次数,添加的(感觉正常来说不需要添加.)
          std::unordered_set<vertex_t> &in_mirror = 
                                            this->supernode_in_mirror[ids_id];
          if(Fc[v] < 0 && ids_id != this->id2spids[u] 
              && in_mirror.find(u) == in_mirror.end()){
            Fc[v] = ids_id;
            this->supernode_source[ids_id].emplace_back(v);
            // build a new spnode idnex
            vid_t supernode_id = 0;
            if (del_index >= 0) {
              supernode_id = delete_spid[del_index]; // 获取被删除的id
              del_index--;
            } else {
              supernode_id = supernodes_num; // 新生成id
              supernodes_num++;
            }
            this->Fc_map[v] = supernode_id;
            {
              std::unique_lock<std::mutex> lk(this->supernode_ids_mux_);
              this->shortcuts[v.GetValue()][ids_id] = supernode_id; // 线程不安全
            }
            this->supernodes[supernode_id].id = v;
            this->supernodes[supernode_id].ids = ids_id;
            this->supernodes[supernode_id].status = false;

            temp_update_source_id.insert(v.GetValue());
            add_num++;
          }
        }
      }
      LOG(INFO) << "  add_time=" << (GetCurrentTime() - add_time);

      double real_del_time = GetCurrentTime();
      for (int i = 0; i <= del_index; i++) {
        this->delete_supernode(delete_spid[i]);
      }
      LOG(INFO) << "  del_index=" << del_index;
      LOG(INFO) << "  real_del_time=" << (GetCurrentTime() - real_del_time);
      
      this->update_cluster_ids.assign(temp_update_cluster_ids.begin(), 
                                       temp_update_cluster_ids.end());
      this->update_source_id.assign(temp_update_source_id.begin(), 
                                     temp_update_source_id.end());

      // debug
      {
        // this->print("inc-compress");
        // for (auto cid : update_cluster_ids) {
        //   LOG(INFO) << " cid=" << cid;
        // }
        // for (auto sid : update_source_id) {
        //   LOG(INFO) << " sid=" << this->vid2Oid(sid);
        // }
      }

      LOG(INFO) << "  spnode_num=" << supernodes_num 
                << " update_source_id.size=" << update_source_id.size();
      LOG(INFO) << "  cluster_num=" << cluster_ids.size() 
                << " update_cluster_ids.size=" << update_cluster_ids.size();
      LOG(INFO) << "  spid rate=" << update_source_id.size()*1.0/supernodes_num;
      LOG(INFO) << "  cid  rate=" << update_cluster_ids.size()*1.0/cluster_ids.size();
      LOG(INFO) << "  add_num=" << add_num << " del_num=" << del_num;
    }

    /**
     * 建立反向的shortcut:
     *  即记录每个cluster内的顶点到入口点的shortcut. 
     *    如下：
     *    shortcuts: entry1 -> {(v1, delta1), (v2, delta2)}
     *            entry2 -> {(v1, delta3)}
     *    reverse_shortcuts: v1 -> {(entry1, delta1), (entry2, delta3)}
     *                     v2 -> {(entry1, delta2)}
     *  reverse_shortcuts: 将包括bound_delta和inner_delta.
     *  其中的依赖的in-mirror作为入口点的已经调整为master的vid.
    */
    void get_reverse_shortcuts() {
      this->reverse_shortcuts.clear();
      this->reverse_shortcuts.resize(this->all_node_num);
      parallel_for(vid_t i = 0; i < this->GetClusterSize(); i++) {
        auto& entry_node_set = this->supernode_source[i];
        for (auto v : entry_node_set) {
          vid_t spid = this->Fc_map[v];
          supernode_t &spnode = this->supernodes[spid];
          for (auto e : spnode.inner_delta) {
            // this->reverse_shortcuts[e.first.GetValue()][v.GetValue()] = e;
            this->reverse_shortcuts[e.first.GetValue()][v.GetValue()] = e.second;
          }
          for (auto e : spnode.bound_delta) {
            this->reverse_shortcuts[e.first.GetValue()][v.GetValue()] = e.second;
          }
        }
        auto& entry_mirror_node_set = this->cluster_in_mirror_ids[i];
        for (auto v : entry_mirror_node_set) {
          vid_t spid = this->Fc_map[v];
          supernode_t &spnode = this->supernodes[spid];
          vid_t master_id = this->mirrorid2vid[v].GetValue();
          for (auto e : spnode.inner_delta) {
            this->reverse_shortcuts[e.first.GetValue()][master_id] = e.second;
          }
          for (auto e : spnode.bound_delta) {
            this->reverse_shortcuts[e.first.GetValue()][master_id] = e.second;
          }
        }
      }
      // debug
      if (false) {
        LOG(INFO) << "-------------------------------------------------------";
        LOG(INFO) << " print reverse shortcut:";
        for (vid_t i = 0; i < this->old_node_num; i++) {
          for (auto& pv : this->reverse_shortcuts[i]) {
            LOG(INFO) << " oid=" << this->vid2Oid(i)
                      << "<-" << this->vid2Oid(pv.first)
                      << " e:" << pv.second;
          }
        }
      }
    }
    
    
    vid_t GetClusterSize() {
      return this->cluster_ids.size();
    }


    // 针对带源点的应用做具体优化：
    //    例如，PHP中，很多cluster不能被源点所作用，故这些cluster对计算没有任
    //   何帮助，于是考虑将这些cluster进行回收，这里的主要目的是避免维护索引
    //   的开销。-- 这里仅仅将被更新但是没有参与过计算的cluster解散。
    void clean_no_used(VertexArray<value_t, vid_t>& values_, value_t default_value) {
      double begin_delete = GetCurrentTime();
      vid_t delete_cluster_num = 0;
      for (auto cid : this->update_cluster_ids) {
        value_t diff = 0;
        for (auto v : this->cluster_ids[cid]) {
          diff += fabs(values_[v] - default_value);
        }
        if (diff == 0) {
          this->delete_cluster(cid);
          delete_cluster_num++;
        }
      }
      LOG(INFO) << "#delete_cluster_num: " << delete_cluster_num;
      LOG(INFO) << "need_update_cluster_num: " 
                << (this->update_cluster_ids.size() - delete_cluster_num);
      LOG(INFO) << "#delete_cluster_time: " << (GetCurrentTime() - begin_delete);
    }

    ~CompressorBase(){
        delete[] supernodes;
    }

public:
    std::shared_ptr<APP_T>& app_;
    std::shared_ptr<fragment_t>& graph_;
    Communicator communicator_;
    CommSpec comm_spec_;
    vid_t supernodes_num=0;
    vid_t MAX_NODE_NUM=FLAGS_max_node_num;
    vid_t MIN_NODE_NUM=FLAGS_min_node_num;
    VertexArray<fc_t, vid_t> Fc; // fc[v]= index of cluster_ids, Fc[v] = ids_id if v is a source node. V doest not include mirror node.
    VertexArray<vid_t, vid_t> Fc_map; // fc[v]= index of supernodes and v is a source node, inclue mirror node, Fc_map[v] = supernode_id;
    supernode_t *supernodes; // max_len = nodes_num
    const vid_t FC_default_value = std::numeric_limits<fc_t>::max(); 
    const vid_t ID_default_value = std::numeric_limits<vid_t>::max(); // max id
    // std::vector<vid_t> supernode_ids;
    std::vector<std::vector<vertex_t>> supernode_ids;  // the set of vertices contained in each supernode
    std::vector<std::vector<vertex_t>> cluster_ids;  // the set of vertices contained in each supernode include mirrorid
    std::vector<std::vector<vertex_t>> supernode_source;  // the set of source vertices of each supernode
    std::vector<std::vector<vertex_t>> cluster_in_mirror_ids;  // the set of mirrorid of in_mirror contained in each supernode include mirror
    std::vector<std::vector<vertex_t>> cluster_out_mirror_ids;  // the set of mirrorid of out_mirror contained in each supernode include mirror
    std::vector<std::unordered_set<vertex_t>> supernode_in_mirror;  // the set of vid of in_mirror vertices of each supernode
    std::vector<std::unordered_set<vertex_t>> supernode_out_mirror;  // the set of vid of out_mirror vertices of each supernode
    std::vector<std::vector<vid_t>> vid2in_mirror_cluster_ids;  // the set of cluster id of each in-mirror vertex
    std::vector<std::vector<vid_t>> vid2in_mirror_mids;  // the set of spid of each in-mirror vertex
    std::vector<std::vector<vid_t>> vid2out_mirror_mids;  // the set of spid of each out-mirror vertex
    // std::vector<std::vector<vid_t>> out_mirror2spids;  // the set of spids of each mirror vertex
    // std::vector<std::vector<vertex_t>> supernode_bound_ids;  // the set of bound vertices of each supernode
    std::vector<short int> supernode_out_bound;  // if is out_bound_node
    VertexArray<vid_t, vid_t> id2spids;                // record the cluster id of each node(include mirror node), note that it is not an index structure id
    std::unordered_set<vid_t> recalculate_spnode_ids;  // the set of recalculated super vertices
    std::unordered_set<vid_t> inccalculate_spnode_ids; // the set of inc-recalculated super vertices
    std::mutex supernode_ids_mux_;
    std::mutex shortcuts_mux_; // for inc_compress
    // std::vector<idx_t> graph_part;  // metis result
    std::vector<std::unordered_map<vid_t, vid_t>> shortcuts; // record shortcuts for each entry vertice (including Mirror vertices)
    std::vector<std::unordered_map<vid_t, delta_t>> reverse_shortcuts; // record re-shortcuts for each entry vertice (including Mirror vertices)
    std::unordered_map<vertex_t, vertex_t> mirrorid2vid; // record the mapping between mirror id and vertex id
    // std::unordered_map<vertex_t, vertex_t> vid2mirrorid; // record the mapping between mirror id and vertex id
    vid_t old_node_num;
    vid_t all_node_num;
    std::vector<std::vector<nbr_t>> subgraph;
    std::vector<std::vector<nbr_t>> subgraph_old;
    std::vector<vid_t> update_cluster_ids; // the set of ids_id of updated cluster
    std::vector<vid_t> update_source_id; // the set of spid of updated supernode
    /* source to in_bound_node */
    Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
    Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
    /* master_source to mirror cluster */
    Array<nbr_index_t, Allocator<nbr_index_t>> im_e_; // in-mirror
    Array<nbr_index_t*, Allocator<nbr_index_t*>> im_e_offset_;
    Array<nbr_index_t, Allocator<nbr_index_t>> om_e_; // out-mirror
    Array<nbr_index_t*, Allocator<nbr_index_t*>> om_e_offset_;
    Array<nbr_index_t, Allocator<nbr_index_t>> oim_e_; // out-mirror
    Array<nbr_index_t*, Allocator<nbr_index_t*>> oim_e_offset_;
    /* in_bound_node to out_bound_node */
    Array<nbr_t, Allocator<nbr_t>> ib_e_;
    Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
    Array<nbr_t, Allocator<nbr_t>> sync_e_; // Synchronized edges between master-mirror without weights.
    Array<nbr_t*, Allocator<nbr_t*>> sync_e_offset_;
    std::vector<vertex_t> all_out_mirror; // all out mirror
    std::vector<vid_t> indegree; // degree of each cluster
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_COMPRESSOR_BASE_H_
