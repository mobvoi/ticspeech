// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#ifndef ENGINE_RESCORER_LM_RESCORER_WRAPPER_H_
#define ENGINE_RESCORER_LM_RESCORER_WRAPPER_H_

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <vector>

#include "base/mru_cache.h"
#include "engine/rescorer/levenshtein_automata.h"
#include "engine/rescorer/lm_rescorer.h"

namespace mobvoi {

class LMRescorerWrapper : RescoringCacheDelegate {
 public:
  explicit LMRescorerWrapper(LMRescorer* rescorer, int cache_size) :
      LMRescorerWrapper(rescorer, false, cache_size) {}

  LMRescorerWrapper(LMRescorer* rescorer, bool own_rescorer, int cache_size) :
      rescorer_(rescorer), own_rescorer_(own_rescorer), app_context_(nullptr) {
    if (cache_size > 0) {
      cache_.reset(new HashingMRUCache<LMHistoryCKey,
           std::pair<float, int>, LMHistoryCKeyHash>(cache_size));
      kenlm_cache_.reset(new HashingMRUCache<RescoringHistoryCKey,
          State, RescoringHistoryCKeyHash>(cache_size));
      if (rescorer_)
        rescorer_->SetRescoringCacheDelegate(this);
    }
  }

  ~LMRescorerWrapper() {
    if (rescorer_)
      rescorer_->SetRescoringCacheDelegate(nullptr);
    if (own_rescorer_) {
      delete rescorer_;
      rescorer_ = nullptr;
    }
  }

  // RescoringCacheDelegate method:
  void OnCacheClear() override;

  template <size_t N>
  float GetLmScore(const RescoringHistoryT<N>& history, int word,
                   int* context_matched = nullptr);
  void Reset();
  // Take ownership of query_context, app_context is not owned here.
  void SetContext(const vector<LabelType>* app_context,
                  vector<LabelType>* query_context) {
    app_context_ = app_context;
    query_context_.reset(query_context);
  }
  void SetContext(const vector<LabelType>* app_context,
                  const vector<int>& keywords_limit) {
    if (app_context != nullptr) BuildLevenAuto(*app_context, keywords_limit);
  }
  float LookAheadScore(uint16_t* state, int word, int* height) const;
  vector<LabelType> GetLevenMatch(const LAState state) const;
  int LookLevenScore(const vector<LAState>& in,
                     const int word,
                     const uint16_t frame,
                     const float score,
                     vector<LAState>* out) const;

 private:
  void BuildAutomaton(const vector<LabelType>& app_context);
  void BuildLevenAuto(const vector<LabelType>& app_context,
                      const vector<int>& keywords_limit);

  LMRescorer* rescorer_;
  bool own_rescorer_;
  const vector<LabelType>* app_context_;
  unique_ptr<vector<LabelType>> query_context_;
  unique_ptr<AhoCorasickTree> tree_;
  unique_ptr<LevenshteinAutomata> leven_auto_;
  int leven_auto_cache_sum_{0};
  // TODO(spye): Consider use HashingHashingMRUCache, compare performance for
  // different cache type.
  unique_ptr<HashingMRUCache<LMHistoryCKey, std::pair<float, int>,
      LMHistoryCKeyHash>> cache_;
  unique_ptr<HashingMRUCache<RescoringHistoryCKey, State,
      RescoringHistoryCKeyHash>> kenlm_cache_;

  DISALLOW_COPY_AND_ASSIGN(LMRescorerWrapper);
};
}  // namespace mobvoi

#endif  // ENGINE_RESCORER_LM_RESCORER_WRAPPER_H_
