// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: spye@mobvoi.com (Shunping Ye)

#include "server/lm/input_predict_server.h"

#include "mobvoi/base/file.h"
#include "mobvoi/base/string_util.h"
#include "mobvoi/base/singleton.h"
#include "engine/rescorer/lm_rescorer_wrapper.h"
#include "grammar/segmentation_utils.h"
#include "third_party/jsoncpp/json.h"

DEFINE_string(model_path,
              "/data/search/graph_clg_20180227/graph/lm_irstlm.9e-10.binary", "");  // NOLINT
DEFINE_int32(ngram_order, 4, "");
DEFINE_string(segmenter_dict, "/data/search/graph_clg_20180227/graph/lexicon", "");  // NOLINT
DEFINE_string(symbol_table, "/data/search/graph_clg_20180227/graph/words.txt", "");  // NOLINT

namespace mobvoi {

InputPredictServer::InputPredictServer() {
  mobvoi::LMRescorerConfig config;
  config.set_model_type("KenLMRescorer");
  auto params = config.add_kenlm_config();
  params->set_ngram_order(FLAGS_ngram_order);
  params->set_model_path(FLAGS_model_path);

  // Segmenter
  segmenter_ = Singleton<DoubleArrayWordSegmenter>::get();
  dynamic_cast<DoubleArrayWordSegmenter*>(segmenter_)->LoadDict(
        FLAGS_segmenter_dict);

  symbol_table_.reset(fst::SymbolTable::ReadText(FLAGS_symbol_table));
  lm_rescorer_core_ = mobvoi::LMRescorer::Create(config.model_type());
  lm_rescorer_core_->Init(config, symbol_table_.get());
  lm_rescorer_.reset(
      new LMRescorerWrapper(lm_rescorer_core_.get(), true, 10000));
}

InputPredictServer::~InputPredictServer() {}

float InputPredictServer::GetScore(const vector<string>& segs,
                                   uint32 word4) {
  if (segs.empty()) {
    mobvoi::RescoringHistory history;
    return lm_rescorer_->GetLmScore(history, word4);
  }

  mobvoi::RescoringHistory history(symbol_table_->Find(segs[0]));
  for (size_t i = 1; i < segs.size(); ++i) {
    mobvoi::RescoringHistory history_tmp;
    uint32 w = symbol_table_->Find(segs[i]);
    mobvoi::RescoringUtil::UpdateHistory(history, w, &history_tmp);
    history = history_tmp;
  }
  return lm_rescorer_->GetLmScore(history, word4);
}

struct Record {
  string word;
  float weight;
};

static string GenResponseForSuggestion(const string& query,
                                       const vector<string>& results) {
  string res;
  res += "XBox.kUpdate(";
  Json::Value n;
  // first element is query.
  // second element is array.
  Json::Value r;
  for (const auto& item : results) {
    Json::Value t;
    t["w"] = item;
    t["k"] = 4;
    r.append(t);
  }
  n["q"] = query;
  n["vs"] = "kv410";
  n["t"] = 0;
  n["r"] = r;
  // TODO(xx) : Append "g" ?
  Json::FastWriter writer;
  res += writer.write(n);
  res += ")";
  return res;
}

bool InputPredictServer::HandleRequest(util::HttpRequest* request,
                                       util::HttpResponse* response) {
  map<string, string> params;
  request->GetQueryParams(&params);
  string word = params["query"];
  LOG(INFO) << "receive word : " << word;
  vector<string> segs;
  segmenter_->Segment(word, &segs);
  LOG(INFO) << "word size : " << segs.size();
  LOG(INFO) << "segmented word list : " << JoinVectorToString(segs, " ");

  auto cmp = [](const Record& left, const Record& right) {
    return (left.weight) < (right.weight);
  };
  std::priority_queue<Record, std::vector<Record>, decltype(cmp)> q(cmp);
  const int kNBest = 10;
  // Save top N words.
  fst::SymbolTableIterator iref(*symbol_table_);
  for (iref.Reset(); !iref.Done(); iref.Next()) {
    uint32 word4 = iref.Value();
    if (word4 == 0) continue;  // skip <eps>
    if (iref.Symbol() == "#0") continue;
    float s = GetScore(segs, word4);
    VLOG(1) << "word : " << iref.Symbol() << ", weight : " << s
            << ", word id : " << word4;
    Record r;
    r.weight = s;
    r.word = iref.Symbol();
    q.push(r);
    if (q.size() > kNBest) {
      q.pop();
    }
  }
  LOG(INFO) << "queue size : " << q.size();
  vector<string> ret;
  while (!q.empty()) {
    auto r = q.top();
    VLOG(1) << "word : " << r.word << ", weight : " << r.weight;
    q.pop();
    ret.push_back(r.word);
  }
  reverse(ret.begin(), ret.end());
  // Try to predict best next word using lm.
  response->AppendHeader("Content-Type", "text/plain; charset=utf-8");
  // Segment.
  response->AppendBuffer(GenResponseForSuggestion(word, ret));
  return true;
}
}  // namespace mobvoi
