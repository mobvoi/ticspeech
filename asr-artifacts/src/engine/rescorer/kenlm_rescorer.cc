// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#include "engine/rescorer/kenlm_rescorer.h"

#include <cmath>
#include <limits>

#include "engine/rescorer/rescorer_model_manager.h"
#include "engine/rescorer/rescoring_utils.h"
#include "mobvoi/base/file.h"
#include "mobvoi/base/hash.h"
#include "mobvoi/base/log.h"
#include "third_party/openfst/include/fst/types.h"

namespace {

const char kRescorerBaseModelGroup[] = "secondpass";

const char* kDefaultEnabledModels[] = {"secondpass", "bugfix", "newword"};

// -std::log(10)
const float kNegLn10 = -2.302585f;

// Weight for OOVs.
const float kOOVRet = -1.5f;
}  // namespace

namespace mobvoi {

KenLMRescorer::KenLMRescorer()
    : rescorer_key_(0), cache_delegate_(nullptr), initialized_(false) {}

KenLMRescorer::~KenLMRescorer() {}

LMRescorer* KenLMRescorer::Copy() const {
  return new KenLMRescorer(*this);
}

void KenLMRescorer::InitInternal(const LMRescorerConfig& config,
                                 const fst::SymbolTable* word_symbols) {
  unique_ptr<RescorerModelManager> manager(new RescorerModelManager("", false));
  manager->Init(config, word_symbols);
  InitInternal(manager.get());
}

void KenLMRescorer::InitInternal(RescorerModelManager* rescorer_model_manager) {
  CHECK(!initialized_);

  // LMRescorer state init.
  SetContextScoringFunctionConfig(rescorer_model_manager->function_config());
  LoadHomophone(rescorer_model_manager->homophone_path());

  // Setup model items.
  models_ = rescorer_model_manager->GetInitModelItems();
  BuildModelIndexes();

  EnableDefaultModels();
}

void KenLMRescorer::BuildModelIndexes() {
  for (size_t i = 0; i < models_.size(); ++i) {
    VLOG(1) << __FUNCTION__ << "model index: " << i
            << ", name: " << models_[i].name
            << ", path: " << models_[i].model_path;
    model_name2id_.emplace(models_[i].name, i);
    model_path2id_.emplace(models_[i].model_path, i);
  }
}

void KenLMRescorer::SyncDynamicModels(
    const RescorerModelManager* rescorer_model_manager) {
  const auto dynamic_models = rescorer_model_manager->GetDynamicModelItems();
  for (auto model_it = dynamic_models.begin(); model_it != dynamic_models.end();
       ++model_it) {
    auto it = model_name2id_.find(model_it->first);
    if (it == model_name2id_.end())
      continue;

    models_[it->second] = *(model_it->second);
  }

  if (cache_delegate_)
    cache_delegate_->OnCacheClear();
}

void KenLMRescorer::EnableModel(int model_index) {
  float weight_sum = models_[model_index].weight;
  for (auto& each : enabled_models_) {
    if (each.first == kRescorerBaseModelGroup)
      continue;
    weight_sum += models_[each.second].weight;
  }

  auto current_model_in_same_group =
      enabled_models_.find(models_[model_index].group);
  if (current_model_in_same_group != enabled_models_.end()) {
    // Already enabled one in same group, disable it first.
    weight_sum -= models_[current_model_in_same_group->second].weight;
    if (weight_sum > 1) {
      LOG(ERROR)
          << "Illegal state: enabled models have sum weights > 1, ignore.";
      return;
    }

    rescorer_key_ &= ~(0x01 << current_model_in_same_group->second);
    current_model_in_same_group->second = model_index;
  } else {
    if (weight_sum > 1) {
      LOG(ERROR)
          << "Illegal state: enabled models have sum weights > 1, ignore.";
      return;
    }

    enabled_models_.emplace(models_[model_index].group, model_index);
  }
  rescorer_key_ |= (0x01 << model_index);
}

void KenLMRescorer::EnableModel(const string& model_name) {
  VLOG(1) << "Enable model: " << model_name;
  auto it = model_name2id_.find(model_name);
  if (it == model_name2id_.end())
    return;
  EnableModel(it->second);
}

void KenLMRescorer::EnableModelByPath(const string& model_path) {
  VLOG(1) << "Enable model: " << model_path;
  auto it = model_path2id_.find(model_path);
  if (it == model_path2id_.end())
    return;
  EnableModel(it->second);
}

void KenLMRescorer::EnableDefaultModels() {
  for (auto i : kDefaultEnabledModels) {
    EnableModel(i);
  }

  // "second_pass" group model must be enabled and have default weight value
  // unchanged.
  auto it = enabled_models_.find(kRescorerBaseModelGroup);
  CHECK(it != enabled_models_.end());
  CHECK(std::fabs(1 - models_[it->second].weight) <
        std::numeric_limits<float>::epsilon());
}

vector<pair<string, string>> KenLMRescorer::GetEnabledModelState() const {
  vector<pair<string, string>> states;
  for (auto it = enabled_models_.begin(); it != enabled_models_.end(); ++it) {
    states.emplace_back(models_[it->second].name,
                        models_[it->second].model_path);
  }
  return states;
}

void KenLMRescorer::GetLogProb(
    const RescoringHistory& history,
    int word,
    RescoringCache* cache,
    float* base_result,
    vector<pair<float, float>>* extra_results) const {
  CHECK(extra_results->empty()) << "Invalid result param.";
  for (auto it = enabled_models_.begin(); it != enabled_models_.end(); ++it) {
    float score = 0.0;
    int32 model_index = it->second;
    if (!models_[model_index].is_valid) {
      continue;
    }
    auto model = models_[model_index].model.get();
    switch (models_[model_index].model_type) {
      case lm::ngram::PROBING:
        score = Query<lm::ngram::ProbingModel>(
            *dynamic_cast<lm::ngram::ProbingModel*>(model),
            model_index, history, word, cache);
        break;
      case lm::ngram::REST_PROBING:
        score = Query<lm::ngram::RestProbingModel>(
            *dynamic_cast<lm::ngram::RestProbingModel*>(model),
            model_index, history, word, cache);
        break;
      case lm::ngram::TRIE:
        score = Query<lm::ngram::TrieModel>(
            *dynamic_cast<lm::ngram::TrieModel*>(model),
            model_index, history, word, cache);
        break;
      case lm::ngram::QUANT_TRIE:
        score = Query<lm::ngram::QuantTrieModel>(
            *dynamic_cast<lm::ngram::QuantTrieModel*>(model),
            model_index, history, word, cache);
        break;
      case lm::ngram::ARRAY_TRIE:
        score = Query<lm::ngram::ArrayTrieModel>(
            *dynamic_cast<lm::ngram::ArrayTrieModel*>(model),
            model_index, history, word, cache);
        break;
      case lm::ngram::QUANT_ARRAY_TRIE:
        score = Query<lm::ngram::QuantArrayTrieModel>(
            *dynamic_cast<lm::ngram::QuantArrayTrieModel*>(model),
            model_index, history, word, cache);
        break;
      default:  // ARPA format
        score = Query<lm::ngram::ProbingModel>(
            *dynamic_cast<lm::ngram::ProbingModel*>(model),
            model_index, history, word, cache);
        break;
    }
    VLOG(3) << "GetLogProb|word: " << word << ", score: "
            << score << ", model: " << model_index;

    if (it->first == kRescorerBaseModelGroup)
      *base_result = score;
    else
      extra_results->emplace_back(models_[model_index].weight, score);
  }
}

template<class Model>
float KenLMRescorer::Query(const Model& model, int32 model_index,
                           const RescoringHistory& history, int word,
                           RescoringCache* cache) const {
  int current_word;
  if (word == dcd::kEndOfSentence) {
    current_word = model.GetVocabulary().EndSentence();
  } else {
    auto it = models_[model_index].relabel_pair->find(word);
    if (it == models_[model_index].relabel_pair->end() || it->second == 0) {
      return kOOVRet;
    } else {
      current_word = it->second;
    }
  }

  lm::ngram::State out{};
  bool history_cached = false;
  // Perhaps using |model_index| to distinguish models within a big cache
  // is not a good idea now because of the mutablitiy of the model itself.
  // We currenlty clear all caches to avoid dirty data.
  // TODO(JIANG Yichen): Do we have any other option? e.g. Cache per model,
  // so that we do not need to clear caches of all models. Also to hide
  // concrete implementation, the |model_index| is decoupled from hash key
  // computation.
  RescoringHistoryCKey cachekey(history, model_index);
  if (cache != nullptr) {
    auto it = cache->Get(cachekey);
    if (it != cache->end()) {
      memcpy(&out, &it->second, sizeof(out));
      history_cached = true;
    }
  }

  if (!history_cached) {
    // cachekey NOT in cache in this branch.
    unsigned int words[kHistoryOrder] = {};
    int count = 0;
    ExtractContext(models_[model_index], history, words, &count);
    lm::ngram::State state =
        count < (models_[model_index].ngram_order - 1) ?
        model.BeginSentenceState() : model.NullContextState();
    int t = count - models_[model_index].ngram_order + 1;
    if (t < 0) t = 0;
    out = state;
    for (; t < count; ++t) {
      lm::WordIndex vocab = words[count - t - 1];
      if (static_cast<LabelType>(vocab) == dcd::kOOVLabel || vocab == 0)
        return kOOVRet;
      model.FullScore(state, vocab, out);
      state = out;
    }
    if (cache != nullptr) {
      State cache_state{};
      memcpy(&cache_state, &state, sizeof(state));
      cache->PutNewKey(cachekey, cache_state);
    }
  }

  lm::ngram::State state;
  auto score = model.FullScore(out, current_word, state).prob * kNegLn10;
  if (cache != nullptr) {
    RescoringHistory new_history;
    RescoringUtil::UpdateHistory(history, word, &new_history);
    State cache_state{};
    memcpy(&cache_state, &state, sizeof(state));

    // Here we don't know whether newkey is in cache or not, so we use Put
    // instead of PutNewKey.
    RescoringHistoryCKey newkey(new_history, model_index);
    cache->Put(newkey, cache_state);
  }
  return score;
}

// Return an id representing the current enabled rescorers, for lm cache key.
int32 KenLMRescorer::GetRescorerKey() const {
  return rescorer_key_;
}

void KenLMRescorer::SetRescoringCacheDelegate(
    RescoringCacheDelegate* delegate) {
  cache_delegate_ = delegate;
}

void KenLMRescorer::ExtractContext(const RescorerModelItem& model,
                                   const RescoringHistory& history,
                                   unsigned int* words,
                                   int* count) const {
  const auto& relabel_pair = model.relabel_pair;
  int32 ngram_order = model.ngram_order;

  int max_history = 2;
  if (history[0] != dcd::kSentenceBoundary) {
    auto it = relabel_pair->find(history[0]);
    words[0] = (it == relabel_pair->end()) ? dcd::kOOVLabel : it->second;
    *count = 1;
  }
  if (max_history >= ngram_order || history[0] == dcd::kSentenceBoundary ||
      history[1] == dcd::kSentenceBoundary) {
    return;
  }
  ++max_history;
  auto it = relabel_pair->find(history[1]);
  words[1] = (it == relabel_pair->end()) ? dcd::kOOVLabel : it->second;
  if (max_history >= ngram_order || history[2] == dcd::kSentenceBoundary) {
    *count = 2;
    return;
  }
  ++max_history;
  it = relabel_pair->find(history[2]);
  words[2] = (it == relabel_pair->end()) ? dcd::kOOVLabel : it->second;
  if (max_history >= ngram_order || history[3] == dcd::kSentenceBoundary) {
    *count = 3;
    return;
  }
  it = relabel_pair->find(history[3]);
  words[3] = (it == relabel_pair->end()) ? dcd::kOOVLabel : it->second;
  *count = 4;
}

REGISTER_LM_RESCORER(KenLMRescorer);
}  // namespace mobvoi
