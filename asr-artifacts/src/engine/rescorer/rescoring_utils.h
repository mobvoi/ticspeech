// Copyright 2018 Mobvoi Inc. All Rights Reserved.
// Author: mingzou@mobvoi.com (Ming Zou)

#ifndef ENGINE_RESCORER_RESCORING_UTILS_H_
#define ENGINE_RESCORER_RESCORING_UTILS_H_

#include <queue>

#include "engine/decoder/constants.h"
#include "fst/types.h"
#include "mobvoi/base/compat.h"

namespace mobvoi {

template<std::size_t N = 4>
struct RescoringHistoryT {
  LabelType words[N]{0};

  RescoringHistoryT() = default;
  explicit RescoringHistoryT(LabelType ilabel) { words[0] = ilabel; }
  RescoringHistoryT(LabelType* p, int n) {
    for (int i = 0; i < order() && i < n; ++i) {
      words[i] = *(p++);
    }
  }

  LabelType operator[](std::size_t idx) const {
    if (idx < static_cast<size_t>(order())) {
      return words[idx] == 0 ? dcd::kSentenceBoundary : words[idx];
    }
    LOG(FATAL) << "Out of range!";
    return words[0];  // Just to make compilers happy
  }

  LabelType& operator[](std::size_t idx) {
    if (idx < order()) return words[idx];
    LOG(FATAL) << "Out of range!";
    return words[0];  // Just to make compilers happy
  }

  bool operator<(const RescoringHistoryT<N>& other) const {
    if (words[0] < other.words[0]) return true;

    for (int i = 1; i < order(); ++i) {
      if (words[i - 1] > other.words[i - 1]) return false;
      if (words[i] < other.words[i]) {
        return true;
      }
    }
    return false;
  }

  bool operator==(const RescoringHistoryT<N>& other) const {
    for (int i = 0; i < order(); ++i) {
      if (words[i] != other.words[i]) return false;
    }
    return true;
  }

  int size() const {
    for (int i = 0; i < order(); ++i) {
      if (words[i] == 0) return i;
    }
    return order();
  }
  constexpr int order() const { return N; }
};

typedef RescoringHistoryT<4> RescoringHistory;
typedef RescoringHistoryT<8> LongRescoringHistory;

// We support at most 5-gram LM rescore.
constexpr int kHistoryOrder = 5;

class RescoringHistoryHash {
 public:
  size_t operator()(const RescoringHistory& rh) const {
    size_t key(0);
    for (int i = 0; i < rh.order() && i < 11; ++i) {
      key += static_cast<size_t>(rh.words[0]) * primes[i];
    }
    return key;
  }

 private:
  int8 primes[11] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
};

class RescoringUtil {
 public:
  template <std::size_t N>
  static void UpdateHistory(const RescoringHistoryT<N>& history,
                            LabelType ilabel,
                            RescoringHistoryT<N>* new_history) {
    if (history.words[0] == dcd::kClassTagIdForRescoring) {  // closing tag
      new_history->words[0] = ilabel;
      return;
    }
    if (ilabel == 0) {
      *new_history = history;
      return;
    }
    for (size_t i = 1; i < N; ++i) {
      new_history->words[i] = history.words[i - 1];
    }
    new_history->words[0] = ilabel;
  }

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(RescoringUtil);
};

// A POD for cache. It will be converted to kenlm/lm/state.hh:State in KenlmRescoreer.  // NOLINT
class State {
 public:
  unsigned int words[kHistoryOrder - 1];
  float backoff[kHistoryOrder - 1];
  unsigned char length;
};

class AhoCorasickTree {
 public:
  void Insert(const vector<LabelType>& ids) {
    Insert(ids.cbegin(), ids.cend());
  }

  template <typename T>
  void Insert(const T& begin, const T& end) {
    int u = 0;
    for (auto id = begin; id != end; ++id) {
      CHECK_LE(*id, kMaxSize) << *id;
      if (!tr_[u][*id]) {
        tr_[u][*id] = ++tot_;
      }
      u = tr_[u][*id];
      CHECK_LE(u, kNumState) << u;
      height_[u] = std::distance(begin, id) + 1;
    }
    end_[u] = true;
  }

  void Build() {
    std::queue<int> q;
    for (int i = 0; i < kMaxSize; i++) {
      if (tr_[0][i]) {
        q.push(tr_[0][i]);
      }
    }
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int i = 0; i < kMaxSize; i++) {
        if (tr_[u][i] > 0) {
          if (!end_[tr_[u][i]]) {
            auto state = fail_[u];
            auto trans = tr_[state][i];
            while (trans == 0 && state != 0) {
              state = fail_[state];
              trans = tr_[state][i];
            }
            fail_[tr_[u][i]] = trans;
          } else {
            fail_[tr_[u][i]] = 0;
          }

          q.push(tr_[u][i]);
        }
      }
    }
  }

  // return height diff.
  int Query(uint16_t* state, uint16_t label) const {
    auto current_state = *state;
    auto s = tr_[current_state][label];
    while (s == 0 && current_state != 0) {
      current_state = fail_[current_state];
      s = tr_[current_state][label];
    }

    *state = s;
    return height_[s];
  }

 private:
  constexpr static int kNumState = 2000;
  constexpr static int kMaxSize = 150;
  uint16_t tr_[kNumState][kMaxSize]{{0}};
  uint16_t tot_{0};
  uint16_t fail_[kMaxSize]{0};
  bool end_[kNumState]{false};
  uint8_t height_[kNumState]{0};
};
}  // namespace mobvoi
#endif  // ENGINE_RESCORER_RESCORING_UTILS_H_
