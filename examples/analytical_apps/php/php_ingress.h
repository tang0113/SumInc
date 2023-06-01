
#ifndef ANALYTICAL_APPS_PHP_PHP_INGRESS_H_
#define ANALYTICAL_APPS_PHP_PHP_INGRESS_H_

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"

namespace grape {

/*
  对于php算法，如果切换了源点，则源点所在的cluster的shortcut需要被重新计算.
    目前针对的是不切换源点的.
*/

template <typename FRAG_T, typename VALUE_T>
class PHPIngress : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_index_t = AdjList<vid_t, value_t>;
  vid_t source_gid;
  vid_t source_vid = std::numeric_limits<vid_t>::max();
  // VertexArray<typename FRAG_T::edata_t, vid_t> weight_sum;
  // VertexArray<value_t, vid_t> weight_sum;
  const value_t d = 0.80f;

  void iterate_begin(FRAG_T& frag) override {
    auto iv = frag.InnerVertices();
    auto source_id = FLAGS_php_source;

    frag.weight_sum.Init(iv, 0);

    for (auto v : iv) {
      auto oes = frag.GetOutgoingAdjList(v);

      for (auto& e : oes) {
        auto dst = e.neighbor;
        auto weight = e.data;

        if (frag.GetId(dst) != source_id) {
          frag.weight_sum[v] += weight;
        }
      }
    }

    // debug: 得到最大出度和最小出度的点oid
    {
      LOG(INFO) << " Just debug:";
      size_t max_id = 0;
      size_t min_id = 0;
      size_t max_odegree = 0;
      size_t min_odegree = std::numeric_limits<size_t>::max();
      for (auto v : iv) {
        auto oes = frag.GetOutgoingAdjList(v);
        if (oes.Size() > max_odegree) {
          max_id = frag.GetId(v);
          max_odegree = oes.Size();
        }
        if (oes.Size() < min_odegree) {
          min_id = frag.GetId(v);
          min_odegree = oes.Size();
        }
      }
      LOG(INFO) << "max_id=" << max_id << " max_odegree=" << max_odegree;
      LOG(INFO) << "min_id=" << min_id << " min_odegree=" << min_odegree;
      // exit(0);
    }
  }

  void rebuild_graph(FRAG_T& frag) override {
    // LOG(INFO) << "\n" << "recover edges' weight...";
    // auto iv = frag.InnerVertices();
    // auto source_id = FLAGS_php_source;

    // for (auto v : iv) {
    //   auto oes = frag.GetOutgoingAdjList(v);

    //   for (auto& e : oes) {
    //     auto dst = e.neighbor;
    //     auto weight = e.data;

    //     if (frag.GetId(dst) != source_id) {
    //       e.data = weight * weight_sum[v];
    //     }
    //   }
    // }
  }

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag) override {
    auto source_id = FLAGS_php_source;
    vertex_t source;
    auto native_source = frag.GetInnerVertex(source_id, source);

    if (native_source) {
      source_vid = source.GetValue();
    }

    CHECK(frag.Oid2Gid(source_id, source_gid));
    if (native_source && source == v) {  // 判断是否是源点
      delta = 1;
    } else {
      delta = 0;
    }
  }

  void init_v(const vertex_t v, value_t& value) override { value = 0.0f; }

  bool accumulate_atomic(value_t& a, value_t b) override {
    atomic_add(a, b);
    return true;
  }

  bool accumulate(value_t& a, value_t b) override {
    a += b;
    return true;
  }

  void priority(value_t& pri, const value_t& value,
                const value_t& delta) override {
    pri = delta;
  }

  void g_function(const FRAG_T& frag, const vertex_t v, const value_t& value,
                  const value_t& delta, const adj_list_t& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.Size();
      auto it = oes.begin();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif
      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto dst = e.neighbor;

        // if (frag.Vertex2Gid(dst) != source_gid) {
        if (dst.GetValue() != source_vid) {
          value_t outv = e.data * delta * d / frag.weight_sum[v];
          this->accumulate_to_delta(dst, outv);
        }
      })
    }
  }

  /* in_bound node send message */
  inline void g_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_t& old_oes, const adj_list_t& to_send_oes) override {
    if (delta != default_v()) {
      auto out_degree = to_send_oes.Size();
      auto it = to_send_oes.begin();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif
      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto dst = e.neighbor;

        // if (frag.Vertex2Gid(dst) != source_gid) {
        if (dst.GetValue() != source_vid) {
          value_t outv = e.data * delta * d / frag.weight_sum[v];
          this->accumulate_to_delta(dst, outv);
        }
      })
    }
  }

  /* source send message */
  inline void g_index_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_index_t& oes, VertexArray<value_t, vid_t>& bound_node_values) override{
    if (delta != default_v()) {
      auto out_degree = oes.Size();
      value_t outv;

      if (out_degree > 0) {
        auto it = oes.begin();
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          // auto dst = e.neighbor;
          // if (frag.Vertex2Gid(dst) != source_gid) {
            this->accumulate_atomic(bound_node_values[e.neighbor], e.data * delta);
          // }
        })
      } 
    }
  }

  inline void g_function(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const adj_list_t& oes, const Nbr<vid_t, edata_t>& oe, value_t& outv) {
    outv = default_v(); // Note: Must ensure that outv must have a return value.
    // if (delta != default_v()) {
      auto dst = oe.neighbor;
      // if (frag.Vertex2Gid(dst) != source_gid) {
      if (dst.GetValue() != source_vid) {
        outv = oe.data * delta * d / frag.weight_sum[v];
      }
    // }
  }

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag, const vertex_t source) override {
    if(v == source){
      delta = 1;
    } 
    else{
      delta = 0;
    }
  }

  void g_revfunction(value_t& value, value_t& rt_value){
      rt_value = value;
  }

  // Used for the interior of the supernode in the later stage of convergence
  inline void g_index_func_delta(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const std::vector<std::pair<vertex_t, value_t>>& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.size();

      if (out_degree > 0) {
        #ifdef COUNT_ACTIVE_EDGE_NUM
          atomic_add(this->f_send_delta_num, (long long)out_degree);
        #endif

        granular_for(j, 0, out_degree, (out_degree > 1024), {
          // auto& e = *(it + j);
          auto& e = oes[j];
          // auto dst = e.first;
          // if (frag.Vertex2Gid(dst) != source_gid) {
            this->accumulate_to_value(const_cast<Vertex<vid_t>&>(e.first), delta * e.second);
          // }
        })
      }  
    }
  }

  inline void g_index_func_value(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, const value_t& delta,
                         const std::vector<std::pair<vertex_t, value_t>>& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.size();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif

      if (out_degree > 0) {
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          // auto& e = *(it + j);
          auto& e = oes[j];
          // auto dst = e.first;
          // if (frag.Vertex2Gid(dst) != source_gid) {
            this->accumulate_to_value(const_cast<Vertex<vid_t>&>(e.first), delta * e.second);
          // }
        })
      }  
    }
  }

  //===============================PULL====================================
  void get_last_delta(const FRAG_T& frag, const std::vector<std::vector<vertex_t>>& all_nodes) override {
    auto inner_vertices = frag.InnerVertices();
    for(int i = 0; i < 4; i++){
      const std::vector<vertex_t>& nodes = all_nodes[i];
      vid_t node_size = nodes.size();
      parallel_for(vid_t j = 0; j < node_size; j++){
        // LOG(INFO) << "i=" << i << "j=" << j;
        vertex_t u = nodes[j];
        auto& delta = this->deltas_[u];
        auto& index_value = this->index_values_[u];
        this->last_deltas_[u] = delta * d; // weight_sum[u];
        this->last_index_values_[u] = index_value;
        delta = this->default_v();
        index_value = this->default_v();
        // LOG(INFO) << "----";
      }
    }
  }

  void get_last_delta(const FRAG_T& frag) override {
    auto inner_vertices = frag.InnerVertices();
    auto size = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
    parallel_for(vid_t j = 0; j < size; j++){
      vertex_t v(j);
      auto& delta = this->deltas_[v];
      this->last_deltas_[v] = delta * d; // / weight_sum[v];
      delta = this->default_v();
    }
  }

  inline void g_function_pull(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_t& ies) override {
    auto in_degree = ies.Size();
    if (in_degree > 0) {
      // if(in_degree <= 1024){
        for(auto ie : ies){
          accumulate(delta, this->last_deltas_[ie.neighbor] * ie.data);
        }
      // }
      // else{
      //   auto it = ies.begin();
      //   cilk::reducer_opadd<value_t> total(delta);
      //   parallel_for(vid_t j = 0; j < in_degree; j++){
      //     auto& ie = *(it + j);
      //     // accumulate(delta, this->last_deltas_[ie.neighbor]);
      //     total += this->last_deltas_[ie.neighbor];
      //   }
      //   delta = total.get_value();
      // }
    } 
  }

  inline void g_function_pull_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes) override {
    auto in_degree = iindexes.Size();
    if (in_degree > 0) {
      for(auto ie : iindexes){
        accumulate(delta, this->last_index_values_[ie.neighbor]*ie.data);
      }
    }
  }

  inline void g_function_pull_spnode_datas_by_index(const FRAG_T& frag, const vertex_t v,
                         const value_t& value, value_t& delta,
                         const adj_list_index_t& iindexes, VertexArray<value_t, vid_t>& spnode_datas) override {
    auto in_degree = iindexes.Size();
    if (in_degree > 0) {
      for(auto ie : iindexes){
        accumulate(delta, spnode_datas[ie.neighbor]*ie.data);
      }
    }
  }
  //========================================================================

  value_t default_v() override { return 0; }

  value_t min_delta() override { return 0; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_PHP_PHP_INGRESS_H_
