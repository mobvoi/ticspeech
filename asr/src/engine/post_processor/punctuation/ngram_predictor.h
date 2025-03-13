// Copyright 2016 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#ifndef ENGINE_PUNCTUATION_NGRAM_PREDICTOR_H_
#define ENGINE_PUNCTUATION_NGRAM_PREDICTOR_H_

#include "engine/post_processor/punctuation/predictor.h"
#include "mobvoi/base/compat.h"
#include "third_party/kenlm/lm/config.hh"
#include "third_party/kenlm/lm/model.hh"
#include "third_party/kenlm/lm/word_index.hh"

namespace mobvoi {

class NgramModel;

class NgramPredictor : public PunctuationPredictor {
 public:
  NgramPredictor(const string& punctuation_model, int punctuation_model_order);
  virtual ~NgramPredictor();

 private:
  virtual const string Predict(const string& sentence) const;

  NgramModel* model_;
  lm::WordIndex period_;
  lm::WordIndex question_mark_;
  int ngram_order_;

  DISALLOW_COPY_AND_ASSIGN(NgramPredictor);
};

}  // namespace mobvoi
#endif  // ENGINE_PUNCTUATION_NGRAM_PREDICTOR_H_
