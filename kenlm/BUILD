licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kenlm",
    srcs = [
        "lm/bhiksha.cc",
        "lm/binary_format.cc",
        "lm/config.cc",
        "lm/lm_exception.cc",
        "lm/model.cc",
        "lm/quantize.cc",
        "lm/read_arpa.cc",
        "lm/search_hashed.cc",
        "lm/search_trie.cc",
        "lm/sizes.cc",
        "lm/trie.cc",
        "lm/trie_sort.cc",
        "lm/value_build.cc",
        "lm/virtual_interface.cc",
        "lm/vocab.cc",
        "util/bit_packing.cc",
        "util/double-conversion/bignum.cc",
        "util/double-conversion/bignum-dtoa.cc",
        "util/double-conversion/cached-powers.cc",
        "util/double-conversion/diy-fp.cc",
        "util/double-conversion/double-conversion.cc",
        "util/double-conversion/fast-dtoa.cc",
        "util/double-conversion/fixed-dtoa.cc",
        "util/double-conversion/strtod.cc",
        "util/ersatz_progress.cc",
        "util/exception.cc",
        "util/file.cc",
        "util/file_piece.cc",
        "util/float_to_string.cc",
        "util/integer_to_string.cc",
        "util/mmap.cc",
        "util/murmur_hash.cc",
        "util/parallel_read.cc",
        "util/pool.cc",
        "util/read_compressed.cc",
        "util/scoped.cc",
        "util/spaces.cc",
        "util/string_piece.cc",
        "util/usage.cc",
    ],
    hdrs = glob([
        "lm/*.hh",
        "util/*.hh",
        "util/double-conversion/*.h",
    ]),
    copts = ["-DKENLM_MAX_ORDER=5"],
    includes = [
        ".",
        "include",
    ],
    deps = [
        "//mobvoi/base:base",
    ],
    linkopts = ["-lrt"],
)

cc_binary(
    name = "build_binary",
    srcs = ["lm/build_binary_main.cc"],
    deps = [":kenlm"],
    copts = ['-DKENLM_MAX_ORDER=5'],
)
