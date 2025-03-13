// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)
// Kenlm arpa binary file has its own vocabulary, so we need to map it to real
// words.txt in decoding.

#include "fst/symbol-table.h"
#include "mobvoi/base/at_exit.h"
#include "mobvoi/base/compat.h"
#include "mobvoi/base/file.h"
#include "mobvoi/base/flags.h"
#include "third_party/kenlm/lm/model.hh"

DEFINE_string(word_symbol_table,
              "",
              "word symbol table for crf model");
DEFINE_string(kenlm_model, "", "interrogative words list");
DEFINE_string(output_file, "", "output file name");

template <class Model>
void GenerateMap() {
  lm::ngram::Config lm_config;
  unique_ptr<Model> model(new Model(FLAGS_kenlm_model.c_str(), lm_config));
  auto symbol_table = fst::SymbolTable::ReadText(FLAGS_word_symbol_table);
  unordered_map<int, int> relabel_pair;
  fst::SymbolTableIterator iref(*symbol_table);
  for (iref.Reset(); !iref.Done(); iref.Next()) {
    relabel_pair[iref.Value()] = model->GetVocabulary().Index(iref.Symbol());
  }
  std::ostringstream oss;
  for (auto iter = relabel_pair.begin(); iter != relabel_pair.end(); ++iter) {
    oss << iter->first << "\t" << iter->second << "\n";
  }
  mobvoi::File::WriteStringToFile(oss.str(), FLAGS_output_file);
}

int main(int argc, char** argv) {
  base::AtExitManager at_exit;
  const string usage =
      "Save map between kenlm binary vocabulary and words.txt\n"
      "Usage:  kenlm_vocab_relabel_main [options] \n";
  mobvoi::ParseCommandLineFlags(&argc, &argv, true, usage);
  lm::ngram::ModelType model_type;
  if (RecognizeBinary(FLAGS_kenlm_model.c_str(), model_type)) {
    switch (model_type) {
      case lm::ngram::PROBING:
        GenerateMap<lm::ngram::ProbingModel>();
        break;
      case lm::ngram::REST_PROBING:
        GenerateMap<lm::ngram::RestProbingModel>();
        break;
      case lm::ngram::TRIE:
        GenerateMap<lm::ngram::TrieModel>();
        break;
      case lm::ngram::QUANT_TRIE:
        GenerateMap<lm::ngram::QuantTrieModel>();
        break;
      case lm::ngram::ARRAY_TRIE:
        GenerateMap<lm::ngram::ArrayTrieModel>();
        break;
      case lm::ngram::QUANT_ARRAY_TRIE:
        GenerateMap<lm::ngram::QuantArrayTrieModel>();
        break;
      default:
        LOG(FATAL) << "Unrecognized kenlm model type " << model_type;
    }
  } else {
    LOG(FATAL) << "Unrecognized kenlm model type";
  }
  return 0;
}
