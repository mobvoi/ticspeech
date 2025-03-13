// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#include "engine/rescorer/lm_rescorer_wrapper.h"

namespace mobvoi {

void LMRescorerWrapper::OnCacheClear() {
  if (cache_) {
    cache_->Clear();
  }
  if (kenlm_cache_) {
    kenlm_cache_->Clear();
  }
}

template <>
float LMRescorerWrapper::GetLmScore(const RescoringHistory& history,
                                    int word, int* context_matched) {
  if (word == 0) {
    return .0f;
  }
  if (history[0] == dcd::kClassTagIdForRescoring) {
    return dcd::kMaxCost;
  }
  if (cache_) {
    LMHistoryCKey lm_cache_key(word, history, rescorer_->GetRescorerKey());

    auto it = cache_->Peek(lm_cache_key);
    if (it != cache_->end()) {
      const auto& pair = it->second;
      if (context_matched != nullptr) {
        *context_matched = pair.second;
      }
      return pair.first;
    }

    auto score = rescorer_->GetLmScore(history, word,
        query_context_ == nullptr ? vector<LabelType>() : *query_context_,
        app_context_ == nullptr ? vector<LabelType>() : *app_context_,
        context_matched, kenlm_cache_.get());
    // Here we use PutNewKey instead of Put, since we know that the key is NOT
    // in the cache, if in it, this function will return in previous branch.
    if (context_matched != nullptr) {
      cache_->PutNewKey(lm_cache_key, std::make_pair(score, *context_matched));
    } else {
      cache_->PutNewKey(lm_cache_key, std::make_pair(score, false));
    }
    return score;
  }
  return rescorer_->GetLmScore(history, word,
      query_context_ == nullptr ? vector<LabelType>() : *query_context_,
      app_context_ == nullptr ? vector<LabelType>() : *app_context_,
      context_matched, kenlm_cache_.get());
}

template <size_t N>
float LMRescorerWrapper::GetLmScore(const RescoringHistoryT<N>& history,
                                    int word, int* context_matched) {
  if (word == 0) {
    return .0f;
  }
  if (history[0] == dcd::kClassTagIdForRescoring) {
    return dcd::kMaxCost;
  }
  return rescorer_->GetLmScore(history, word,
      query_context_ == nullptr ? vector<LabelType>() : *query_context_,
      app_context_ == nullptr ? vector<LabelType>() : *app_context_,
      context_matched);
}

void LMRescorerWrapper::Reset() {
  if (cache_) {
    cache_->Clear();
  }
  if (kenlm_cache_) {
    kenlm_cache_->Clear();
  }
  query_context_.reset();
  tree_.reset();
  app_context_ = nullptr;
}

void LMRescorerWrapper::BuildAutomaton(const vector<LabelType>& app_context) {
  tree_.reset(new AhoCorasickTree());
  if (app_context.empty()) return;
  int start = app_context.size() - 1;
  for (int i = app_context.size() - 1; i >= 0; --i) {
    if (app_context[i] == dcd::kSentenceBoundary) {
      if (i < start - 1) {
        tree_->Insert(app_context.crbegin() + app_context.size() - start,
                      app_context.crbegin() + app_context.size() - i - 1);
      }
      start = i;
    }
  }
  tree_->Build();
}

void LMRescorerWrapper::BuildLevenAuto(const vector<LabelType>& app_context,
                                       const vector<int>& keywords_limit) {
  if (leven_auto_cache_sum_ == std::accumulate(app_context.begin(),
                                              app_context.end(),
                                              0)) {
    LOG(INFO) << "Levenshtein automata: Cache";
    return;
  }
  leven_auto_.reset(new LevenshteinAutomata());
  if (app_context.empty()) return;
  int start = app_context.size() - 1;
  vector<LabelType> word;
  int distance = 1;
  int limit_index = keywords_limit.size() - 1;
  while (start > 0) {
    while (app_context[start] == dcd::kSentenceBoundary) start--;
    word.clear();
    while (app_context[start] != dcd::kSentenceBoundary) {
      word.emplace_back(app_context[start--]);
    }
    int limit = (limit_index >= 0) ? keywords_limit[limit_index--] : 0;
    leven_auto_->Insert(word, distance, limit);
  }
  leven_auto_->Build();
  leven_auto_cache_sum_ = std::accumulate(app_context.begin(),
                                          app_context.end(),
                                          0);
  LOG(INFO) << "Levenshtein automata: Build";
}

float LMRescorerWrapper::LookAheadScore(uint16_t* state, int word,
                                        int* height) const {
  if (!tree_) {
    LOG(FATAL) << "Please build automaton first!";
  }
  return rescorer_->LookAheadScore(*tree_, state, word, height);
}

vector<LabelType> LMRescorerWrapper::GetLevenMatch(const LAState state) const {
  return leven_auto_->KeywordMatch(state);
}

int LMRescorerWrapper::LookLevenScore(const vector<LAState>& in,
                                      const int word,
                                      const uint16_t frame,
                                      const float score,
                                      vector<LAState>* out) const {
  if (!leven_auto_) {
    LOG(FATAL) << "Please build levenshtein automata first!";
  }
  leven_auto_->ExpandStep(in, word, frame, score, out);
  int height = 0;
  for (auto state : *out) {
    if (state.height > height) height = state.height;
  }
  return height;
}

template float LMRescorerWrapper::GetLmScore(
    const LongRescoringHistory& history, int word, int* context_matched);
}  // namespace mobvoi
