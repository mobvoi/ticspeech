// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#ifndef ENGINE_RESCORER_KENLM_RESCORER_H_
#define ENGINE_RESCORER_KENLM_RESCORER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "engine/rescorer/lm_rescorer.h"
#include "third_party/kenlm/lm/model.hh"

namespace mobvoi {

struct RescorerModelItem {
  bool is_valid = false;
  int32 ngram_order = 4;
  string model_path;
  shared_ptr<unordered_map<LabelType, LabelType>> relabel_pair;
  shared_ptr<lm::base::Model> model;
  lm::ngram::ModelType model_type = lm::ngram::PROBING;
  float weight = 1.0f;
  string group;
  string name;
};

class KenLMRescorer: public LMRescorer {
 public:
  KenLMRescorer();
  virtual ~KenLMRescorer();
  KenLMRescorer(const KenLMRescorer& rhs) = default;

  // LMRescorer methods:
  LMRescorer* Copy() const override;
  int32 GetRescorerKey() const override;
  void SetRescoringCacheDelegate(RescoringCacheDelegate* delegate) override;
  void EnableModel(const string& model_name) override;
  void EnableModelByPath(const string& model_path) override;
  void SyncDynamicModels(
      const RescorerModelManager* rescorer_model_manager) override;
  vector<pair<string, string>> GetEnabledModelState() const override;

 private:
  void BuildModelIndexes();
  void EnableDefaultModels();

  void EnableModel(int model_index);

  void ExtractContext(const RescorerModelItem& model,
                      const RescoringHistory& history,
                      unsigned int* words,
                      int* count) const;

  template <class Model>
  float Query(const Model& model,
              int32 model_index,
              const RescoringHistory& history,
              int word,
              RescoringCache* cache) const;

  // LMRescorer methods:
  void InitInternal(const LMRescorerConfig& config,
                    const fst::SymbolTable* word_symbols) override;
  void InitInternal(RescorerModelManager* rescorer_model_manager) override;

  void GetLogProb(const RescoringHistory& history,
                  int word,
                  RescoringCache* cache,
                  float* base_result,
                  vector<pair<float, float>>* extra_results) const override;
  void GetLogProb(const LongRescoringHistory& history,
                  int word,
                  RescoringCache* cache,
                  float* base_result,
                  vector<pair<float, float>>* extra_results) const override {
    LOG(FATAL) << "only 4-gram kenlm is supported!";
  }

  // Override operator:
  KenLMRescorer& operator=(const KenLMRescorer& rhs);

  // All available models
  vector<RescorerModelItem> models_;

  // Name search index.
  map<string, int> model_name2id_;

  // This is kept for compatibility. We used to identify a model by path,
  // but we use model name now.
  // TODO(JIANG Yichen): Check if we can change poi selector to use model name.
  // The advantage is that the poi location mapping has no need to change even
  // if the model path is updated. Then EnbaleModelByPath() can be removed.
  map<string, int> model_path2id_;

  // Record currently enabled models. Only one model can be enabled in the same
  // group. |enabled_models_| maintains a mapping from group name to enabled
  // model.
  map<string, int> enabled_models_;

  // This is used for supplementary cache key. It is the bit mask of the
  // enabled lm rescorder models in |enabled_models_|. Currently, we support
  // 32 models in max (including the base rescorer).
  int32 rescorer_key_;

  RescoringCacheDelegate* cache_delegate_;

  // The lm models are shared among rescorer instances. Init() should be invoked
  // only once.
  bool initialized_;
};

}  // namespace mobvoi

#endif  // ENGINE_RESCORER_KENLM_RESCORER_H_
