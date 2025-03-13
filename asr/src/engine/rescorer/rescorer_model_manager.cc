// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: yichen.jiang@mobvoi.com (JIANG Yichen)

#include "engine/rescorer/rescorer_model_manager.h"

#include <vector>
#include <map>

#include "engine/rescorer/lm_rescorer.h"
#include "mobvoi/base/file.h"
#include "mobvoi/base/file/proto_util.h"
#include "mobvoi/base/log.h"
#include "third_party/openfst/include/fst/types.h"
#include "third_party/kenlm/lm/model.hh"

namespace {

const char kDynamicConfigFileName[] = "config.proto";

const char kRescorerBaseModelName[] = "secondpass";
const char kRescorerBaseModelGroup[] = "secondpass";

const char kRescorerBugfixModelName[] = "bugfix";
const char kRescorerBugfixModelGroup[] = "bugfix";

const char kRescorerNewWordModelName[] = "newword";
const char kRescorerNewWordModelGroup[] = "newword";

const char kRescorerPoiModelName[] = "poi";
const char kRescorerPoiModelGroup[] = "poi";

const unordered_set<string> kGroupsSupportingDynamicConfig = {
    kRescorerBugfixModelGroup, kRescorerNewWordModelGroup};
}  // namespace

namespace mobvoi {

RescorerModelManager::UpdateRequest::UpdateRequest(
    RescorerModelManager* manager,
    const KenLMConfig& static_config,
    const string& dynamic_config_path,
    shared_ptr<RescorerModelItem> item)
    : manager_(manager),
      merged_config_(static_config),
      dynamic_config_path_(dynamic_config_path),
      model_name_(item->name),
      item_(item) {}

RescorerModelManager::UpdateRequest::~UpdateRequest() {}

bool RescorerModelManager::UpdateRequest::DoUpdate(const KenLMConfig& config) {
  VLOG(1) << __FUNCTION__ << " model name: " << model_name_ << " write to "
          << dynamic_config_path_;
  if (!WriteProtoToFile(dynamic_config_path_, config)) {
    LOG(ERROR) << "Failed to update dynamic config proto file "
               << dynamic_config_path_;
    return false;
  }

  merged_config_.MergeFrom(config);
  manager_->UpdateModelItem(merged_config_, item_.get());
  if (!item_->is_valid) {
    LOG(WARNING) << "Update model: " << model_name_ << " to invalid state";
  }
  return true;
}

void RescorerModelManager::UpdateRequest::Commit() {
  manager_->CommitUpdate(this);
}

RescorerModelManager::RescorerModelManager(const string& recognizer_model_dir,
                                           bool enable_dynamic_config)
    : model_base_dir_(recognizer_model_dir),
      word_symbols_(nullptr),
      dynamic_config_enabled_(enable_dynamic_config) {}

void RescorerModelManager::Init(const LMRescorerConfig& config,
                                const fst::SymbolTable* word_symbols) {
  LOG(INFO) << "config file epoch is: " << config.epoch()
            << ", config base dir is: " << model_base_dir_;

  homophone_path_ = config.homophone_path();
  function_config_ = config.function_config();
  word_symbols_ = word_symbols;

  if (config.epoch() == 1) {
    ParseLMRescorerConfigV1(config);
  } else {
    ParseLMRescorerConfig(config);
  }
}

RescorerModelManager::~RescorerModelManager() {}

void RescorerModelManager::ParseLMRescorerConfigV1(
    const LMRescorerConfig& config) {
  LMRescorerConfig rewrite_config(config);
  for (int i = 0; i < rewrite_config.kenlm_config_size(); ++i) {
    KenLMConfig* rewrite_kenlm_config = rewrite_config.mutable_kenlm_config(i);
    if (i >= config.base_score_config().weight_size()) {
      if (i == 0) {
        rewrite_kenlm_config->set_weight(1.0f);
      } else {
        rewrite_kenlm_config->set_weight(0.0f);
        LOG(WARNING) << "kenlm_config (" << i << ") is configured without base "
                                                 "score config weight, set its "
                                                 "weight to 0";
      }
    } else {
      rewrite_kenlm_config->set_weight(config.base_score_config().weight(i));
    }

    if (i == 0) {
      rewrite_kenlm_config->set_name(kRescorerBaseModelName);
      rewrite_kenlm_config->set_group(kRescorerBaseModelGroup);
    } else if (i == 1) {
      rewrite_kenlm_config->set_name(kRescorerBugfixModelName);
      rewrite_kenlm_config->set_group(kRescorerBugfixModelGroup);
    } else {
      rewrite_kenlm_config->set_name(kRescorerPoiModelName + std::to_string(i));
      rewrite_kenlm_config->set_group(kRescorerPoiModelGroup);
    }
  }
  ParseLMRescorerConfig(rewrite_config);
}

void RescorerModelManager::ParseLMRescorerConfig(
    const LMRescorerConfig& config) {
  for (int i = 0; i < config.kenlm_config_size(); ++i) {
    const KenLMConfig& kenlm_config = config.kenlm_config(i);
    RescorerModelItem item;
    LoadModelItem(kenlm_config, &item);
    VLOG(1) << __FUNCTION__ << " name = " << item.name
            << ", group = " << item.group
            << ", ngram_order = " << item.ngram_order
            << ", weight = " << item.weight << ", path = " << item.model_path;

    init_models_.emplace(item.name, item);

    if (dynamic_config_enabled_ &&
        (kGroupsSupportingDynamicConfig.find(item.group) !=
         kGroupsSupportingDynamicConfig.end())) {
      supporting_dynamic_models_.emplace(item.name, kenlm_config);
    }
  }
}

void RescorerModelManager::LoadModelItem(const KenLMConfig& static_config,
                                         RescorerModelItem* item) {
  KenLMConfig merged_config(static_config);
  KenLMConfig dynamic_config;
  string dynamic_config_path = File::JoinPath(
      model_base_dir_,
      File::JoinPath(static_config.name(), kDynamicConfigFileName));
  if (dynamic_config_enabled_ && File::Exists(dynamic_config_path) &&
      ReadProtoFromFile(dynamic_config_path, &dynamic_config)) {
    // Dynamic config override static config.
    merged_config.MergeFrom(dynamic_config);
  }
  UpdateModelItem(merged_config, item);
}

void RescorerModelManager::UpdateModelItem(const KenLMConfig& config,
                                           RescorerModelItem* item) {
  item->name = config.name();
  item->group = config.group();
  item->weight = config.weight();
  item->ngram_order = config.ngram_order();
  if (item->model_path == config.model_path()) {
    // Model file is not changed, so we can ignore the expensive operations.
    // |is_valid| state also keeps unchanged.
    return;
  }
  item->model_path = config.model_path();
  if (!File::Exists(item->model_path)) {
    item->is_valid = false;
    return;
  }

  VLOG(1) << "Load model: " << item->name
          << " begin, model_path = " << item->model_path;
  item->model.reset(lm::ngram::LoadVirtual(
      item->model_path.c_str(), lm::ngram::Config(), item->model_type));
  VLOG(1) << "Load model: " << item->name << " done.";

  VLOG(1) << "Generate relabel mapping for " << item->name << " begin.";
  item->relabel_pair.reset(
      CreateRelabelMapping(item->model_type, item->model.get()));
  VLOG(1) << "Generate relabel mapping for " << item->name << " done.";

  item->is_valid = (item->model.get() && item->relabel_pair.get());
}

unordered_map<LabelType, LabelType>* RescorerModelManager::CreateRelabelMapping(
    const lm::ngram::ModelType model_type,
    const lm::base::Model* model) {
  switch (model_type) {
    case lm::ngram::PROBING:
      return RelabelMappingHelper<lm::ngram::ProbingModel>(
          dynamic_cast<const lm::ngram::ProbingModel*>(model));
    case lm::ngram::REST_PROBING:
      return RelabelMappingHelper<lm::ngram::RestProbingModel>(
          dynamic_cast<const lm::ngram::RestProbingModel*>(model));
    case lm::ngram::TRIE:
      return RelabelMappingHelper<lm::ngram::TrieModel>(
          dynamic_cast<const lm::ngram::TrieModel*>(model));
    case lm::ngram::QUANT_TRIE:
      return RelabelMappingHelper<lm::ngram::QuantTrieModel>(
          dynamic_cast<const lm::ngram::QuantTrieModel*>(model));
    case lm::ngram::ARRAY_TRIE:
      return RelabelMappingHelper<lm::ngram::ArrayTrieModel>(
          dynamic_cast<const lm::ngram::ArrayTrieModel*>(model));
    case lm::ngram::QUANT_ARRAY_TRIE:
      return RelabelMappingHelper<lm::ngram::QuantArrayTrieModel>(
          dynamic_cast<const lm::ngram::QuantArrayTrieModel*>(model));
    default:
      LOG(FATAL) << "Unrecognized kenlm model type " << model_type;
  }
  return nullptr;
}

template <class Model>
unordered_map<LabelType, LabelType>* RescorerModelManager::RelabelMappingHelper(
    const Model* model) {
  auto relabel_pair = new unordered_map<LabelType, LabelType>();
  fst::SymbolTableIterator iref(*word_symbols_);
  for (iref.Reset(); !iref.Done(); iref.Next()) {
    relabel_pair->emplace(
        iref.Value(),
        static_cast<LabelType>(model->GetVocabulary().Index(iref.Symbol())));
  }
  return relabel_pair;
}

vector<RescorerModelItem> RescorerModelManager::GetInitModelItems() const {
  vector<RescorerModelItem> models;
  for (auto it = init_models_.begin(); it != init_models_.end(); ++it) {
    models.push_back(it->second);
  }
  return models;
}

const map<string, shared_ptr<RescorerModelItem>>&
RescorerModelManager::GetDynamicModelItems() const {
  return dynamic_models_;
}

bool RescorerModelManager::UpdateModel(const string& model_name,
                                       const KenLMConfig& dynamic_config) {
  shared_ptr<UpdateRequest> update = MakeUpdateRequest(model_name);
  if (!update)
    return false;

  if (!update->DoUpdate(dynamic_config))
    return false;

  update->Commit();
  return true;
}

shared_ptr<RescorerModelManager::UpdateRequest>
RescorerModelManager::MakeUpdateRequest(const string& model_name) {
  shared_ptr<UpdateRequest> request;
  auto index = supporting_dynamic_models_.find(model_name);
  if (index == supporting_dynamic_models_.end()) {
    LOG(WARNING) << "Cannot find " << model_name
                 << " in dynamic model supporting list, is it a wrong name?";
    return request;
  }

  shared_ptr<RescorerModelItem> item;
  auto it = dynamic_models_.find(model_name);
  if (it == dynamic_models_.end()) {
    // |dynamic_models_| is lazily loaded because usually most of the models
    // supporing dynamic update remain unchanged.
    item.reset(new RescorerModelItem(init_models_[model_name]));
  } else {
    item.reset(new RescorerModelItem(*it->second));
  }

  const string dynamic_config_path = File::JoinPath(
      model_base_dir_, File::JoinPath(model_name, kDynamicConfigFileName));
  request.reset(new RescorerModelManager::UpdateRequest(
      this, index->second, dynamic_config_path, item));
  return request;
}

void RescorerModelManager::CommitUpdate(
    RescorerModelManager::UpdateRequest* update_request) {
  auto it = dynamic_models_.find(update_request->model_name());
  if (it == dynamic_models_.end()) {
    dynamic_models_.emplace(update_request->model_name(),
                            update_request->rescorer_model_item());
  } else {
    it->second = update_request->rescorer_model_item();
  }
}

}  // namespace mobvoi
