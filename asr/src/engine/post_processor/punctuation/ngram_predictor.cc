// Copyright 2016 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#include "engine/post_processor/punctuation/ngram_predictor.h"

#include "mobvoi/base/log.h"
#include "mobvoi/base/macros.h"
#include "mobvoi/base/singleton.h"
#include "mobvoi/base/string_util.h"
#include "third_party/kenlm/lm/enumerate_vocab.hh"
#include "third_party/kenlm/lm/return.hh"
#include "third_party/kenlm/lm/state.hh"

DEFINE_string(punctuation_model, "", "ngram punctuation model.");
DEFINE_int32(punctuation_model_order, 4, "order of punctuation model.");

namespace mobvoi {

// Singleton ngram model.
class NgramModel {
 public:
  ~NgramModel() {}

  static string punctuation_model_;
  static void InitParams(const string& punctuation_model) {
    punctuation_model_ = punctuation_model;
  }

  lm::ngram::Model* model() {
    return model_.get();
  }

 private:
  NgramModel();
  friend struct DefaultSingletonTraits<NgramModel>;
  unique_ptr<lm::ngram::Model> model_;

  DISALLOW_COPY_AND_ASSIGN(NgramModel);
};

string NgramModel::punctuation_model_ = "";

NgramModel::NgramModel() {
  lm::ngram::Config config;
  CHECK(!punctuation_model_.empty()) << "No model is found!";
  model_.reset(new lm::ngram::Model(punctuation_model_.c_str(), config));
}

NgramPredictor::NgramPredictor(const string& punctuation_model,
                               int punctuation_model_order)
    : ngram_order_(punctuation_model_order) {
  NgramModel::InitParams("");
  model_ = Singleton<NgramModel>::get();
  period_ = model_->model()->GetVocabulary().Index("。");
  question_mark_ = model_->model()->GetVocabulary().Index("?");
}

NgramPredictor::~NgramPredictor() {}

const string NgramPredictor::Predict(const std::string& sentence) const {
  bool sentence_context = false;

  lm::ngram::State state = sentence_context ? model_->model()->BeginSentenceState()
                                            : model_->model()->NullContextState();
  lm::ngram::State out;
  vector<string> words;
  mobvoi::SplitStringToVector(sentence, " ", true, &words);

  size_t count = words.size();
  size_t t = count - ngram_order_ + 1;
  if (t < 0) t = 0;
  for (; t < count; ++t) {
    lm::WordIndex vocab = model_->model()->GetVocabulary().Index(words[t]);
    model_->model()->FullScore(state, vocab, out);
    state = out;
  }
  double period_score = model_->model()->FullScore(state, period_, out).prob;
  double question_score = model_->model()->FullScore(state, question_mark_, out).prob;
  return period_score  > question_score ? kPeriodMark : kQuestionMark;
}

}  // namespace mobvoi
