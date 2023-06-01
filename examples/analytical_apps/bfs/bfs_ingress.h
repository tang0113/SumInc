
#ifndef ANALYTICAL_APPS_BFS_BFS_INGRESS_H_
#define ANALYTICAL_APPS_BFS_BFS_INGRESS_H_

#include "flags.h"
#include "grape/app/traversal_app_base.h"
#include "grape/util.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class BFSIngress : public TraversalAppBase<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = typename TraversalAppBase<FRAG_T, VALUE_T>::value_t;
  using delta_t = typename TraversalAppBase<FRAG_T, VALUE_T>::delta_t;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using adj_list_index_t = AdjList<vid_t, delta_t>;
  // value_t imax = std::numeric_limits<value_t>::max();

  value_t GetInitValue(const vertex_t& v) const override {
    return GetIdentityElement();
    // return imax;
  }

  delta_t GetInitDelta(const vertex_t& v) const override {
    vertex_t source;
    bool native_source =
        this->fragment().GetInnerVertex(FLAGS_sssp_source, source);
    value_t init_dist = GetIdentityElement();

    if (native_source && source == v) {
      init_dist = 0;
    }

    return this->GenDelta(v, init_dist);
  }

  bool CombineValueDelta(value_t& lhs, const delta_t& rhs) {
    if (lhs > rhs.value) {
      lhs = rhs.value;
      return true;
    }
    return false;
  }

  bool AccumulateDeltaAtomic(delta_t& lhs, const delta_t& rhs) override {
    return lhs.SetIfLessThanAtomic(rhs);
  }

  value_t GetPriority(const vertex_t& v, const value_t& value,
                      const delta_t& delta) const override {
    return delta.value;
  }

  void Compute(const vertex_t& u, const value_t& value, const delta_t& delta,
               DenseVertexSet<vid_t>& modified) override {
    auto dist = delta.value;
    auto oes = this->fragment().GetOutgoingAdjList(u);

    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        auto new_dist = 1 + dist;
        // LOG(INFO) << "vid:" << u.GetValue()
        //           << " ->" << v.GetValue()
        //           << " w=" << delta.value;
        if (this->deltas_[v].value > new_dist) {
          delta_t delta_to_send = this->GenDelta(u, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          modified.Insert(v);
          // this->touch_nodes.Insert(v);
          //atomic_add(this->node_update_num, (long long)1);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        auto new_dist = 1 + dist;
        if (this->deltas_[v].value > new_dist) {
          delta_t delta_to_send = this->GenDelta(u, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          modified.Insert(v);
        }
      }
    }
  }

  value_t GetIdentityElement() const override {
    return std::numeric_limits<value_t>::max();
  }

  // to support compress
  bool AccumulateDelta(delta_t& lhs, const delta_t& rhs) override {
    return lhs.SetIfLessThan(rhs);
  }

  delta_t GetInitDelta(const vertex_t& v, const vertex_t& source) const override {
    value_t init_dist = GetIdentityElement();

    if (source == v) {
      init_dist = 0;
    }

    return this->GenDelta(v, init_dist);
  }

  void Compute(const vertex_t& v, const value_t& value, const delta_t& delta,
                const adj_list_t& oes, const Nbr<vid_t, edata_t>& oe, delta_t& outv) override {
    auto new_dist = delta.value + 1;
    outv = this->GenDelta(v, new_dist);
    //atomic_add(this->f_send_delta_num, (long long)1); // // 注意为了不在此处发生线程阻塞，放到了iter_commpress.h里面
  }

  void Compute(const vertex_t& u, const value_t& value, const delta_t& delta,
               const adj_list_t& oes,
               DenseVertexSet<vid_t>& modified) override {
    auto dist = delta.value;
    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        auto new_dist = 1 + dist;
        // LOG(INFO) << "vid:" << u.GetValue()
        //           << " ->" << v.GetValue()
        //           << " w=" << delta.value;
        if (this->deltas_[v].value > new_dist) {
          delta_t delta_to_send = this->GenDelta(u, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          modified.Insert(v);
          // this->touch_nodes.Insert(v);
          //atomic_add(this->node_update_num, (long long)1);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        auto new_dist = 1 + dist;
        delta_t delta_to_send = this->GenDelta(u, new_dist);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      }
    }
  }

  void ComputeByIndexDelta(const vertex_t& u, const value_t& value, const delta_t& delta,
               const std::vector<std::pair<vertex_t, delta_t>>& oes,
               DenseVertexSet<vid_t>& modified) override {
    auto dist = delta.value;
    // this->f_send_delta_num += oes.size();

    if (FLAGS_cilk) {
      auto out_degree = oes.size();
      auto it = oes.begin();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif
      
      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.first;
        auto new_dist = e.second.value + dist;
        // LOG(INFO) << "vid:" << u.GetValue()
        //           << " ->" << v.GetValue()
        //           << " w=" << delta.value;
        if (this->deltas_[v].value > new_dist) {
          delta_t delta_to_send = this->GenDelta(e.second.parent_gid, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          // modified.Insert(v); // 注意:如果仅仅是入口点最后校正阶段时调用本函数，则无须记录活跃点，因为已经收敛了.
          // this->CombineValueDelta(this->values_[v], this->deltas_[v]); // 注意:这个仅仅入口点最后校正阶段时能合并!!!
          atomic_min(this->values_[v], new_dist); // 注意:这个仅仅入口点最后校正阶段时能合并!!! 存在多个入口同时写
          // this->touch_nodes.Insert(v);
          //atomic_add(this->node_update_num, (long long)1);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.first;
        auto new_dist = e.second.value + dist;
        if (this->deltas_[v].value > new_dist) {
          delta_t delta_to_send = this->GenDelta(e.second.parent_gid, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          modified.Insert(v);
        }
      }
    }
  }

  void ComputeByIndexDelta(const vertex_t& u, const value_t& value, const delta_t& delta,
               const adj_list_index_t& oes,
               DenseVertexSet<vid_t>& modified) override {
    auto dist = delta.value;
    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();
      #ifdef COUNT_ACTIVE_EDGE_NUM
        atomic_add(this->f_send_delta_num, (long long)out_degree);
      #endif
      
      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        auto new_dist = e.data.value + dist;
        // LOG(INFO) << "vid:" << v.GetValue()
        //           << " ->" << e.neighbor.GetValue()
        //           << " w=" << delta.value;
        if (this->deltas_[v].value > new_dist) { 
          delta_t delta_to_send = this->GenDelta(e.data.parent_gid, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          modified.Insert(v);
          // this->touch_nodes.Insert(v);
          //atomic_add(this->node_update_num, (long long)1);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        auto new_dist = e.data.value + dist;
        if (this->deltas_[v].value > new_dist) { 
          delta_t delta_to_send = this->GenDelta(e.data.parent_gid, new_dist);
          this->AccumulateToAtomic(v, delta_to_send);
          modified.Insert(v);
        }
      }
    }
  }
 
  void revCompute(delta_t& delta, delta_t& rt_delta) override {
    rt_delta = delta;
  }

};

}  // namespace grape

#endif  // ANALYTICAL_APPS_BFS_BFS_INGRESS_H_
