// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#include "engine/rescorer/lm_rescorer.h"

#include "engine/rescorer/lm_rescorer_config.pb.h"
#include "engine/rescorer/rescorer_model_manager.h"
#include "engine/rescorer/rescoring_utils.h"
#include "mobvoi/base/at_exit.h"
#include "mobvoi/base/compat.h"
#include "mobvoi/base/file/proto_util.h"
#include "third_party/gmock/include/gmock/gmock.h"
#include "third_party/gtest/gtest.h"

namespace mobvoi {

class KenLMRescorerTest : public ::testing::Test {
 protected:
  unique_ptr<mobvoi::LMRescorer> rescorer_;

  int CountSetBits(int32 n) {
    int count = 0;
    while (n) {
      n = n & (n - 1);
      count++;
    }
    return count;
  }

  std::pair<std::string, std::string> FindModelInStates(
      const std::vector<std::pair<std::string, std::string>>& states,
      const string& name) {
    for (auto each : states) {
      if (each.first == name) {
        return each;
      }
    }
    return std::make_pair("not found", "not found");
  }
};

class MockRescoringCacheDelegate : public RescoringCacheDelegate {
 public:
  MOCK_METHOD0(OnCacheClear, void());
};

TEST_F(KenLMRescorerTest, Basic) {
  base::AtExitManager at_exit;
  int ngram_order = 4;
  string model_path = "engine/rescorer/testdata/lm.bin";
  string homophone_path = "engine/rescorer/testdata/homophone.bin";
  mobvoi::LMRescorerConfig config;
  config.set_model_type("KenLMRescorer");
  auto params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  config.set_homophone_path(homophone_path);
  rescorer_ = mobvoi::LMRescorer::Create(config.model_type());

  string sentence = "打电话 给 曲 飞";
  int word1 = 28633;
  int word2 = 22801;
  int word3 = 32411;
  int word4 = 10906;
  unique_ptr<fst::SymbolTable> word_symbols(new fst::SymbolTable());
  word_symbols->AddSymbol("打电话", word1);
  word_symbols->AddSymbol("给", word2);
  word_symbols->AddSymbol("曲", word3);
  word_symbols->AddSymbol("飞", word4);
  rescorer_->Init(config, word_symbols.get());

  mobvoi::RescoringHistory history(word1);
  mobvoi::RescoringHistory history_tmp;
  mobvoi::RescoringUtil::UpdateHistory(history, word2, &history_tmp);
  mobvoi::RescoringUtil::UpdateHistory(history_tmp, word3, &history);
  vector<LabelType> context;
  auto score = rescorer_->GetLmScore(history, word4, context, context);

  // Kenlm query output for "打电话 给 曲 飞":
  // 打电话=17251 1 -3.103959	给=27437 1 -2.68515	曲=19576 1 -3.702745 	飞=34698 1 -3.761936	</s>=20 1 -1.592141	Total: -14.845931 OOV: 0  NOLINT
  // ln(10^-3.761936) = -8.6621777544
  EXPECT_FLOAT_EQ(score, 8.6621777f);
}

TEST_F(KenLMRescorerTest, EnableModel) {
  base::AtExitManager at_exit;
  int ngram_order = 4;
  string model_path = "engine/rescorer/testdata/lm.bin";
  string homophone_path = "engine/rescorer/testdata/homophone.bin";
  mobvoi::LMRescorerConfig config;
  config.set_model_type("KenLMRescorer");
  config.set_homophone_path(homophone_path);
  config.set_epoch(2);
  auto params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("secondpass");
  params->set_group("secondpass");
  params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("bugfix");
  params->set_group("bugfix");
  params->set_weight(0.1);
  params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("newword");
  params->set_group("newword");
  params->set_weight(0.1);
  params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("poi_ok1");
  params->set_group("poi");
  params->set_weight(0.5);
  params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("poi_ok2");
  params->set_group("poi");
  params->set_weight(0.5);
  params = config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("poi_err");
  params->set_group("poi");
  params->set_weight(0.99);

  rescorer_ = mobvoi::LMRescorer::Create(config.model_type());
  unique_ptr<fst::SymbolTable> word_symbols(new fst::SymbolTable());
  rescorer_->Init(config, word_symbols.get());

  // Check defaults.
  vector<pair<string, string>> enabled_models =
      rescorer_->GetEnabledModelState();
  EXPECT_EQ(static_cast<int>(enabled_models.size()), 3);
  auto state = FindModelInStates(enabled_models, "secondpass");
  EXPECT_EQ(state.first, "secondpass");
  EXPECT_EQ(state.second, model_path);
  state = FindModelInStates(enabled_models, "bugfix");
  EXPECT_EQ(state.first, "bugfix");
  EXPECT_EQ(state.second, model_path);
  state = FindModelInStates(enabled_models, "newword");
  EXPECT_EQ(state.first, "newword");
  EXPECT_EQ(state.second, model_path);
  int32 key = rescorer_->GetRescorerKey();
  EXPECT_EQ(CountSetBits(key), 3);

  // Enable models.
  rescorer_->EnableModel("not_existed");
  enabled_models = rescorer_->GetEnabledModelState();
  EXPECT_EQ(static_cast<int>(enabled_models.size()), 3);
  EXPECT_EQ(key, rescorer_->GetRescorerKey());

  rescorer_->EnableModel("poi_err");
  enabled_models = rescorer_->GetEnabledModelState();
  EXPECT_EQ(static_cast<int>(enabled_models.size()), 3);
  EXPECT_EQ(key, rescorer_->GetRescorerKey());

  rescorer_->EnableModel("poi_ok1");
  enabled_models = rescorer_->GetEnabledModelState();
  EXPECT_EQ(static_cast<int>(enabled_models.size()), 4);
  int32 key1 = rescorer_->GetRescorerKey();
  EXPECT_EQ(CountSetBits(key1), 4);

  rescorer_->EnableModel("poi_ok2");
  enabled_models = rescorer_->GetEnabledModelState();
  EXPECT_EQ(static_cast<int>(enabled_models.size()), 4);
  int32 key2 = rescorer_->GetRescorerKey();
  EXPECT_EQ(CountSetBits(key2), 4);
  EXPECT_EQ(CountSetBits(key1 ^ key2), 2);
  EXPECT_EQ(CountSetBits(key1 & key2), 3);
  EXPECT_EQ(CountSetBits(key1 | key2), 5);
  state = FindModelInStates(enabled_models, "poi_ok2");
  EXPECT_EQ(state.first, "poi_ok2");
  EXPECT_EQ(state.second, model_path);
  state = FindModelInStates(enabled_models, "poi_ok1");
  EXPECT_EQ(state.first, "not found");
  EXPECT_EQ(state.second, "not found");
}

TEST_F(KenLMRescorerTest, DynamicModel) {
  base::AtExitManager at_exit;
  int ngram_order = 4;
  string model_base_dir = "engine/rescorer/testdata/";
  string model_path = model_base_dir + "lm.bin";
  string homophone_path = model_base_dir + "homophone.bin";

  mobvoi::LMRescorerConfig static_config;
  static_config.set_model_type("KenLMRescorer");
  static_config.set_homophone_path(homophone_path);
  static_config.set_epoch(2);
  auto params = static_config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("secondpass");
  params->set_group("secondpass");
  params = static_config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("bugfix");
  params->set_group("bugfix");
  params->set_weight(0.1);
  params = static_config.add_kenlm_config();
  params->set_ngram_order(ngram_order);
  params->set_model_path(model_path);
  params->set_name("newword");
  params->set_group("newword");
  params->set_weight(0.2);

  KenLMConfig dynamic_config;
  dynamic_config.set_ngram_order(ngram_order);
  dynamic_config.set_model_path("dynamic_model_path");
  dynamic_config.set_name("newword");
  dynamic_config.set_group("newword");
  dynamic_config.set_weight(0.4);
  WriteProtoToFile(model_base_dir + "newword/config.proto", dynamic_config);

  rescorer_ = mobvoi::LMRescorer::Create(static_config.model_type());
  unique_ptr<fst::SymbolTable> word_symbols(new fst::SymbolTable());
  unique_ptr<RescorerModelManager> model_manager(
      new RescorerModelManager(model_base_dir, true));
  model_manager->Init(static_config, word_symbols.get());
  rescorer_->Init(model_manager.get());
  MockRescoringCacheDelegate mock_delegate;
  rescorer_->SetRescoringCacheDelegate(&mock_delegate);

  auto enabled_models = rescorer_->GetEnabledModelState();
  auto state = FindModelInStates(enabled_models, "newword");
  EXPECT_EQ(state.second, "dynamic_model_path");

  dynamic_config.Clear();
  dynamic_config.set_model_path("invalid_path_for_test");
  EXPECT_TRUE(model_manager->UpdateModel("newword", dynamic_config));
  EXPECT_CALL(mock_delegate, OnCacheClear()).Times(2);
  rescorer_->SyncDynamicModels(model_manager.get());

  enabled_models = rescorer_->GetEnabledModelState();
  state = FindModelInStates(enabled_models, "newword");
  EXPECT_EQ(state.second, "invalid_path_for_test");

  dynamic_config.Clear();
  EXPECT_TRUE(model_manager->UpdateModel("newword", dynamic_config));
  rescorer_->SyncDynamicModels(model_manager.get());

  enabled_models = rescorer_->GetEnabledModelState();
  state = FindModelInStates(enabled_models, "newword");
  EXPECT_EQ(state.second, model_path);
}

}  // namespace mobvoi
