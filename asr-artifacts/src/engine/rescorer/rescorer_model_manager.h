// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: yichen.jiang@mobvoi.com (JIANG Yichen)

#ifndef ENGINE_RESCORER_RESCORER_MODEL_MANAGER_H_
#define ENGINE_RESCORER_RESCORER_MODEL_MANAGER_H_

#include <memory>

#include "engine/rescorer/lm_rescorer_config.pb.h"
#include "engine/rescorer/kenlm_rescorer.h"

namespace mobvoi {

class RescorerModelManager {
 public:
  class UpdateRequest {
   public:
    UpdateRequest(RescorerModelManager* manager,
                  const KenLMConfig& static_config,
                  const string& dynamic_config_path,
                  shared_ptr<RescorerModelItem> item);
    ~UpdateRequest();

    bool DoUpdate(const KenLMConfig& config);
    void Commit();

    shared_ptr<RescorerModelItem> rescorer_model_item() const { return item_; }
    const string& model_name() const { return model_name_; }

   private:
    RescorerModelManager* manager_;
    KenLMConfig merged_config_;
    const string dynamic_config_path_;
    const string model_name_;
    shared_ptr<RescorerModelItem> item_;
  };

  RescorerModelManager(const string& recognizer_model_dir,
                       bool enable_dynamic_config);
  virtual ~RescorerModelManager();

  void Init(const LMRescorerConfig& config,
            const fst::SymbolTable* word_symbols);
  vector<RescorerModelItem> GetInitModelItems() const;
  const map<string, shared_ptr<RescorerModelItem>>& GetDynamicModelItems()
      const;

  // Update dynamic config.
  bool UpdateModel(const string& model_name, const KenLMConfig& dynamic_config);

  // UpdateModel() is splitted to three-staged update because of
  // mutex optimization consideration.
  // Note that change will just take effect after re-init unless
  // calling UpdateRequest::Commit() or
  // RescorerModelManager::CommitUpdate()
  shared_ptr<RescorerModelManager::UpdateRequest> MakeUpdateRequest(
      const string& model_name);
  void CommitUpdate(RescorerModelManager::UpdateRequest* update_request);

  const string& homophone_path() const { return homophone_path_; }
  const ContextScoringFunctionConfig& function_config() const {
    return function_config_;
  }

 private:
  // Parse config file in V1 format.
  // TODO(JIANG Yichen): Remove this when ensure no traffic of V1 config.
  void ParseLMRescorerConfigV1(const LMRescorerConfig& config);
  void ParseLMRescorerConfig(const LMRescorerConfig& config);

  // Load real model according to config.
  void LoadModelItem(const KenLMConfig& kenlm_config, RescorerModelItem* item);

  void UpdateModelItem(const KenLMConfig& config, RescorerModelItem* item);

  void RealtimeUpdateModel(RescorerModelItem* item);

  // Base ASR model and rescore models do not share label index. Need to
  // compute mapping between kenlm binary vocabulary and base model symbol
  // table.
  unordered_map<LabelType, LabelType>* CreateRelabelMapping(
      const lm::ngram::ModelType model_type,
      const lm::base::Model* model);
  template <class Model>
  unordered_map<LabelType, LabelType>* RelabelMappingHelper(const Model* model);

  // Model base dir for dynamic model storage.
  const string model_base_dir_;

  // Config reading usge.
  ContextScoringFunctionConfig function_config_;
  string homophone_path_;

  // Store model items for init parsing.
  map<string, RescorerModelItem> init_models_;

  // Keep the latest changed dynamic models for rescorer copies to sync.
  map<string, shared_ptr<RescorerModelItem>> dynamic_models_;

  // Supporting dynamic model update list: model name and static config.
  unordered_map<string, const KenLMConfig> supporting_dynamic_models_;

  // Base ASR model symbol table.
  const fst::SymbolTable* word_symbols_;

  bool dynamic_config_enabled_;

  DISALLOW_COPY_AND_ASSIGN(RescorerModelManager);
};

}  // namespace mobvoi

#endif  // ENGINE_RESCORER_RESCORER_MODEL_MANAGER_H_
