add_definitions(
    -DMATRIX_FULL
)
enable_language(ASM)

set(COMMON_ENGINE_SRC_LIST
  ${ENGINE_SRC_DIR}/post_processor/itn/inverse_text_normalizer.cc
  ${ENGINE_SRC_DIR}/post_processor/noise_filt/lm_embedding.cc
  ${ENGINE_SRC_DIR}/post_processor/noise_filt/noise_filter.cc
  ${ENGINE_SRC_DIR}/post_processor/post_processor_factory.cc
  ${ENGINE_SRC_DIR}/rescorer/kenlm_rescorer.cc
  ${ENGINE_SRC_DIR}/rescorer/rescorer_model_manager.cc
  ${ENGINE_SRC_DIR}/rescorer/rescorer_prototype_wrapper.cc
)

# Note: Files in ${COMMON_ENGINE_SRC_LIST} are excluded here
set(RECOGNIZER_SRC
  ${ENGINE_SRC_DIR}/decoder/lattice-faster-decoder.cc
  ${ENGINE_SRC_DIR}/decoder/lattice-faster-online-decoder.cc
  ${ENGINE_SRC_DIR}/decoder/clg_decoder.cc
  ${ENGINE_SRC_DIR}/decoder/dcd_search_arc.cc
  ${ENGINE_SRC_DIR}/decoder/dcd_search_state.cc
  ${ENGINE_SRC_DIR}/decoder/search_options.cc
  ${ENGINE_SRC_DIR}/hmm/hmm-topology.cc
  ${ENGINE_SRC_DIR}/hmm/hmm_transition_model.cc
  ${ENGINE_SRC_DIR}/hmm/hmm-utils.cc
  ${ENGINE_SRC_DIR}/hmm/kaldi-transition-model.cc
  ${ENGINE_SRC_DIR}/lattice/determinize-lattice-pruned.cc
  ${ENGINE_SRC_DIR}/lattice/kaldi-lattice.cc
  ${ENGINE_SRC_DIR}/lattice/lattice.cc
  ${ENGINE_SRC_DIR}/lattice/lattice-functions.cc
  ${ENGINE_SRC_DIR}/lattice/minimize-lattice.cc
  ${ENGINE_SRC_DIR}/lattice/push-lattice.cc
  ${ENGINE_SRC_DIR}/nnet3/am-nnet-simple.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-analyze.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-common.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-compile-utils.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-compile.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-component-itf.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-computation-graph.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-computation.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-compute.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-descriptor.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-graph.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-nnet.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-optimize-utils.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-optimize.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-parse.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-simple-component.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet-utils.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet3_decodable.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet_cache.cc
  ${ENGINE_SRC_DIR}/nnet3/nnet_stats.cc
  ${ENGINE_SRC_DIR}/one/one_acoustic_model.cc
  ${ENGINE_SRC_DIR}/one/one_decodable.cc
  ${ENGINE_SRC_DIR}/online/graph_manager/cascade.cc
  ${ENGINE_SRC_DIR}/online/graph_manager/clg_manager.cc
  ${ENGINE_SRC_DIR}/online/graph_manager/hclg_manager.cc
  ${ENGINE_SRC_DIR}/online/graph_manager/relabel_file.cc
  ${ENGINE_SRC_DIR}/online/graph_manager/rnnt_manager.cc
  ${ENGINE_SRC_DIR}/online/online-faster-decoder.cc
  ${ENGINE_SRC_DIR}/online/online_clg_decoder.cc
  ${ENGINE_SRC_DIR}/online/online_recognizer.cc
  ${ENGINE_SRC_DIR}/online/online_recognizer_impl.cc
  ${ENGINE_SRC_DIR}/online/online_recognizer_util.cc
  ${ENGINE_SRC_DIR}/online/onlinebin-util.cc
  ${ENGINE_SRC_DIR}/online/recognizer_model.cc
  ${ENGINE_SRC_DIR}/online/recognizer_model_impl.cc
  ${ENGINE_SRC_DIR}/rnnt/online_rnnt_decoder.cc
  ${ENGINE_SRC_DIR}/rnnt/rnnt_decoder_util.cc
  ${ENGINE_SRC_DIR}/rnnt/rnnt_one_acoustic_model.cc
  ${ENGINE_SRC_DIR}/rnnt/rnnt_ort_acoustic_model.cc
  ${ENGINE_SRC_DIR}/rescorer/levenshtein_automata.cc
  ${ENGINE_SRC_DIR}/rescorer/lm_rescorer.cc
  ${ENGINE_SRC_DIR}/rescorer/lm_rescorer_wrapper.cc
  ${ENGINE_SRC_DIR}/seq2seq/context_graph.cc
  ${ENGINE_SRC_DIR}/tree/context-dep.cc
  ${ENGINE_SRC_DIR}/tree/event-map.cc
  ${ENGINE_SRC_DIR}/util/engine_util.cc
  ${ENGINE_SRC_DIR}/util/remapped_embedding.cc
  ${ENGINE_SRC_DIR}/util/pinyin_segmenter.cc
  ${ENGINE_SRC_DIR}/util/poi_model_selector.cc
  ${ENGINE_SRC_DIR}/vad/dnn_voice_detector.cc
  ${ENGINE_SRC_DIR}/vad/voice_event_detector.cc
  ${ENGINE_SRC_DIR}/vad/voice_event_detector_wrapper.cc
  ${MOBVOI_SRC_DIR}/matrix/fast_blas.cc  # TODO(ynlin): put it here since specialization code is in .cc
  ${MOBVOI_SRC_DIR}/matrix/fast_blas_extra.cc  # TODO(ynlin): put it here since specialization code is in .cc
  ${PROJECT_SOURCE_DIR}/../base/thread_pool.cc
  ${GRAMMAR_SRC_DIR}/chinese_number_converter.cc
  ${GRAMMAR_SRC_DIR}/text_normalizer.cc
  ${GRAMMAR_SRC_DIR}/text_normalizer_impl.cc
  ${GRAMMAR_SRC_DIR}/chinese_converter.cc
  ${GRAMMAR_SRC_DIR}/chinese_converter_impl.cc
  ${GRAMMAR_SRC_DIR}/compiler_resource_manager.cc
  ${GRAMMAR_SRC_DIR}/grammar_compiler.cc
  ${GRAMMAR_SRC_DIR}/grammar_compiler_impl.cc
  ${GRAMMAR_SRC_DIR}/grammar_compiler_manager.cc
  ${GRAMMAR_SRC_DIR}/grammar_util.cc
  ${GRAMMAR_SRC_DIR}/multi_dict_trie.cc
  ${GRAMMAR_SRC_DIR}/segmentation_utils.cc
  ${PROJECT_SOURCE_DIR}/../pron/dict_pron_generator.cc
  ${PROJECT_SOURCE_DIR}/../pron/fst_path_finder.cc
  ${PROJECT_SOURCE_DIR}/../pron/g2p_pron_generator.cc
  ${PROJECT_SOURCE_DIR}/../pron/g2p_pron_generator_impl.cc
  ${PROJECT_SOURCE_DIR}/../pron/pron_cache.cc
  ${PROJECT_SOURCE_DIR}/build/engine/online/proto/graph_resource_config.pb.cc
  ${PROJECT_SOURCE_DIR}/build/engine/online/recognizer_params.pb.cc
  ${PROJECT_SOURCE_DIR}/build/engine/online/wav_data.pb.cc
  ${PROJECT_SOURCE_DIR}/build/engine/post_processor/post_processor.pb.cc
  ${PROJECT_SOURCE_DIR}/build/engine/rescorer/lm_rescorer_config.pb.cc
  ${PROJECT_SOURCE_DIR}/build/engine/util/poi_model_path_config.pb.cc
  ${PROJECT_SOURCE_DIR}/build/grammar/proto/compiler_resource_params.pb.cc
  ${PROJECT_SOURCE_DIR}/build/grammar/proto/grammar_compiler_params.pb.cc
  ${PROJECT_SOURCE_DIR}/build/pron/proto/pron_cache.pb.cc
  #${PROJECT_SOURCE_DIR}/build/third_party/onnxruntime-1.0.0/onnxruntime/core/protobuf/onnx-ml.pb.cc
  #${PROJECT_SOURCE_DIR}/build/third_party/onnxruntime-1.0.0/onnxruntime/core/protobuf/onnx-operators-ml.pb.cc
)

set(RECOGNIZER_ALWAYSLINK_STATIC_SRC
  ${SERVER_SRC_DIR}/config_center_container.cc
  ${ENGINE_SRC_DIR}/rescorer/dummy_rescorer.cc
)

set(QUERY_CORRECTOR_SRC
  ${SERVER_SRC_DIR}/query_corrector/poi_corrector.cc
  ${SERVER_SRC_DIR}/query_corrector/poi_tagger.cc
  ${SERVER_SRC_DIR}/query_corrector/query_trie.cc
  ${SERVER_SRC_DIR}/query_corrector/query_darts.cc
  ${PROJECT_SOURCE_DIR}/build/server/query_corrector/query_corrector_config.pb.cc
  ${PROJECT_SOURCE_DIR}/build/server/query_corrector/query_entity.pb.cc
)

list(APPEND RECOGNIZER_SRC ${QUERY_CORRECTOR_SRC})
#add_library(query_corrector STATIC ${QUERY_CORRECTOR_SRC})
#target_link_libraries(query_corrector mobvoi_recognizer_common_static)
#add_executable(query_corrector_main ${SERVER_SRC_DIR}/query_corrector/query_corrector_main)

add_library(lib_rec_objs OBJECT ${RECOGNIZER_SRC})
add_library(lib_common_engine_objs OBJECT ${COMMON_ENGINE_SRC_LIST})

set(JNI_SRC ${PROJECT_BINARY_DIR}/recognizer_wrap.cxx)

if (${OS} STREQUAL "ios" OR ${OS} STREQUAL "mac")
  add_library(mobvoi_recognizer SHARED
    ${RECOGNIZER_SRC}
    $<TARGET_OBJECTS:lib_rec_objs>
    $<TARGET_OBJECTS:lib_common_engine_objs>
  )

  add_library(mobvoi_recognizer_static STATIC $<TARGET_OBJECTS:lib_rec_objs>)
  target_link_libraries(mobvoi_recognizer_static mobvoi_recognizer_common_static)

  add_library(mobvoi_recognizer_alwayslink_static STATIC
    ${RECOGNIZER_ALWAYSLINK_STATIC_SRC}
    $<TARGET_OBJECTS:lib_common_engine_objs>
  )
  target_link_libraries(mobvoi_recognizer_alwayslink_static mobvoi_recognizer_common_static)
elseif (${OS} STREQUAL "android")
  add_library(mobvoi_recognizer SHARED
    $<TARGET_OBJECTS:lib_rec_objs>
    $<TARGET_OBJECTS:lib_common_engine_objs>
  )
  if (ANDROID_ENABLE_GPROF)
    target_link_libraries(mobvoi_recognizer ${NDK_PROFILER_LIB})
  endif ()
  target_link_libraries(mobvoi_recognizer dl log)
  add_library(mobvoi_recognizer_static STATIC $<TARGET_OBJECTS:lib_rec_objs>)
  target_link_libraries(mobvoi_recognizer_static mobvoi_recognizer_common_static)
  add_library(mobvoi_recognizer_alwayslink_static STATIC
    ${RECOGNIZER_ALWAYSLINK_STATIC_SRC}
    $<TARGET_OBJECTS:lib_common_engine_objs>
  )
  target_link_libraries(mobvoi_recognizer_alwayslink_static mobvoi_recognizer_common_static)
elseif (${OS} STREQUAL "linux")
  add_library(mobvoi_recognizer SHARED
    $<TARGET_OBJECTS:lib_rec_objs>
    $<TARGET_OBJECTS:lib_common_engine_objs>
    ${JNI_SRC}
  )
  target_link_libraries(mobvoi_recognizer dl pthread ${BLAS_LIBRARY})
elseif ((${OS} STREQUAL "embedded_linux") OR (${OS} STREQUAL "qnx"))
  add_library(mobvoi_recognizer SHARED
    $<TARGET_OBJECTS:lib_rec_objs>
    $<TARGET_OBJECTS:lib_common_engine_objs>
  )

  add_library(mobvoi_recognizer_static STATIC $<TARGET_OBJECTS:lib_rec_objs>)
  target_link_libraries(mobvoi_recognizer_static mobvoi_recognizer_common_static)

  add_library(mobvoi_recognizer_alwayslink_static STATIC
    ${RECOGNIZER_ALWAYSLINK_STATIC_SRC}
    $<TARGET_OBJECTS:lib_common_engine_objs>
  )
  target_link_libraries(mobvoi_recognizer_alwayslink_static mobvoi_recognizer_common_static)
else()
  MESSAGE("UNKNOWN OS TYPE ${OS}")
endif ()

target_link_libraries(mobvoi_recognizer mobvoi_recognizer_common_static)

if (NOT ${OS} STREQUAL "ios")
  target_link_libraries(mobvoi_recognizer mobvoi_recognizer_common_static)

  if (${OS} STREQUAL "embedded_linux")
    add_executable(symbol_table_bench
      ${MOBVOI_SRC_DIR}/base/time.cc
      ${MOBVOI_SRC_DIR}/base/time_posix.cc
      ${GRAMMAR_SRC_DIR}/symbol_table_bench.cc
      )
    target_link_libraries(symbol_table_bench mobvoi_recognizer_static)
  endif ()

  add_executable(convert_vad_config_main
    ${ENGINE_SRC_DIR}/rnnt/convert_vad_config_main.cc)
  target_link_libraries(convert_vad_config_main mobvoi_recognizer_static)

  add_executable(batch_online_recognizer_main
    ${ENGINE_SRC_DIR}/online/batch_online_recognizer_main.cc)
  target_link_libraries(batch_online_recognizer_main mobvoi_recognizer_static
      -Wl,--whole-archive mobvoi_recognizer_alwayslink_static -Wl,--no-whole-archive)

  add_executable(query_corrector_main ${SERVER_SRC_DIR}/query_corrector/query_corrector_main.cc)
  target_link_libraries(query_corrector_main mobvoi_recognizer_static)

  if ((${OS} STREQUAL "embedded_linux") AND ((${ARCH} STREQUAL "x86_64") OR (${ARCH} STREQUAL "arm64") OR (${ARCH} STREQUAL "arm64-icas3")))
    message("-- Link Tcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    target_link_libraries(batch_online_recognizer_main tcmalloc_static -Wl,--undefined=IsHeapProfilerRunning)
    target_link_libraries(query_corrector_main tcmalloc_static -Wl,--undefined=IsHeapProfilerRunning)
  endif()

  if ((${OS} STREQUAL "embedded_linux") AND (${ARCH} STREQUAL "x86_64") AND (${CMAKE_BUILD_TYPE} STREQUAL "Release"))
    include_directories(${PROJECT_SOURCE_DIR}/../third_party/websocket)
    include_directories(${PROJECT_SOURCE_DIR}/../third_party/breakpad)

    # Required by http_handler.cc for support of Blue/Green deployment
    set(BCRYPT_FILE_LIST
      ${MOBVOI_SRC_DIR}/util/crypto/bcrypt.cc
      ${PROJECT_SOURCE_DIR}/../third_party/bcrypt/bcrypt.c
      ${PROJECT_SOURCE_DIR}/../third_party/bcrypt/crypt_blowfish.c
      ${PROJECT_SOURCE_DIR}/../third_party/bcrypt/crypt_gensalt.c
      ${PROJECT_SOURCE_DIR}/../third_party/bcrypt/wrapper.c)

    add_executable(offline_recognizer_main
      ${BCRYPT_FILE_LIST}
      ${ENGINE_SRC_DIR}/online/app_context_cache.cc
      ${PROJECT_SOURCE_DIR}/build/server/proto/metrics_raw_data.pb.cc
      ${PROJECT_SOURCE_DIR}/build/server/proto/recognizer_config.pb.cc
      ${PROJECT_SOURCE_DIR}/build/server/proto/offline_model.pb.cc
      ${GRAMMAR_SRC_DIR}/grammar_util.cc
      ${PROJECT_SOURCE_DIR}/../server/model_cache/compressed_symbol_table.cc
      ${MOBVOI_SRC_DIR}/util/net/util.cc
      ${MOBVOI_SRC_DIR}/base/breakpad.cc
      ${MOBVOI_SRC_DIR}/base/string_util.cc
      ${MOBVOI_SRC_DIR}/base/time.cc
      ${MOBVOI_SRC_DIR}/base/time_posix.cc
      ${MOBVOI_SRC_DIR}/util/net/email/email_client.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/data_provider.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/event_loop.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/http_handler.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/http_request.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/http_response.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/http_server.cc
      ${MOBVOI_SRC_DIR}/util/net/http_server/http_type.cc
      ${MOBVOI_SRC_DIR}/util/crypto/uuid.cc
      ${MOBVOI_SRC_DIR}/codec/speex_wrapper.cc
      ${MOBVOI_SRC_DIR}/util/wave/sampling_rate_converter.cc
      ${SERVER_SRC_DIR}/util/timer_util.cc
      ${SERVER_SRC_DIR}/util/audio_converter.cc
      ${SERVER_SRC_DIR}/util/get_side_silence.cc
      ${SERVER_SRC_DIR}/util/email_util.cc
      ${SERVER_SRC_DIR}/asr_server.cc
      ${SERVER_SRC_DIR}/deploy_manager.cc
      ${SERVER_SRC_DIR}/connections_metrics_reporter.cc
      ${SERVER_SRC_DIR}/latency_metrics_reporter.cc
      ${SERVER_SRC_DIR}/message_processor.cc
      ${SERVER_SRC_DIR}/metrics_reporter.cc
      ${SERVER_SRC_DIR}/recognizer_event_handler.cc
      ${SERVER_SRC_DIR}/query_corrector/query_corrector_client.cc
      ${SERVER_SRC_DIR}/recognizer_server_main.cc)
    target_link_libraries(offline_recognizer_main
      ${PROJECT_SOURCE_DIR}/../third_party/boost_1_53_0/libs/lib64_release/libboost_filesystem.a
      ${PROJECT_SOURCE_DIR}/../third_party/boost_1_53_0/libs/lib64_release/libboost_program_options.a
      ${PROJECT_SOURCE_DIR}/../third_party/boost_1_53_0/libs/lib64_release/libboost_random.a
      ${PROJECT_SOURCE_DIR}/../third_party/boost_1_53_0/libs/lib64_release/libboost_system.a
      ${PROJECT_SOURCE_DIR}/../third_party/boost_1_53_0/libs/lib64_release/libboost_thread.a
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_intel_lp64.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_core.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_vml_def.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_avx.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_avx2.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_def.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_rt.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_intel_thread.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libiomp5.so
      ${PROJECT_SOURCE_DIR}/../third_party/mkl/lib64_release/libmkl_mc3.so
      ${PROJECT_SOURCE_DIR}/../third_party/breakpad/lib64_release/libbreakpad.a
      mobvoi_recognizer)
  endif ()

  add_executable(online_recognizer_main
    ${ENGINE_SRC_DIR}/online/online_recognizer_main.cc)
  target_link_libraries(online_recognizer_main mobvoi_recognizer_static mobvoi_recognizer_alwayslink_static)

  add_executable(grammar_compiler_main
    ${GRAMMAR_SRC_DIR}/grammar_compiler_main.cc)
  target_link_libraries(grammar_compiler_main mobvoi_recognizer_static mobvoi_recognizer_alwayslink_static)

  if (NOT ${OS} STREQUAL "qnx")
    # include all test except in qnx OS
    file(GLOB_RECURSE test_src_simple
      "${ENGINE_SRC_DIR}/*/*_test.cc"
      "${GRAMMAR_SRC_DIR}/*_test.cc"
    )
    set(test_src_exclude
      ${test_src_exclude}
      "${ENGINE_SRC_DIR}/seq2seq/ctc_prefix_beam_search_test.cc"
      "${ENGINE_SRC_DIR}/decoder/compose_cache_test.cc"
      "${ENGINE_SRC_DIR}/decoder/expander_fst_test.cc"
      "${ENGINE_SRC_DIR}/end_of_sentence/eos_predictor_test.cc"
      "${ENGINE_SRC_DIR}/end_of_sentence/eos_rule_checker_test.cc"
      "${ENGINE_SRC_DIR}/gmm/diag_gmm_decodable_test.cc"
      "${ENGINE_SRC_DIR}/frontend/delaysum_beamformer_test.cc"
      "${ENGINE_SRC_DIR}/frontend/feature_extraction_util_test.cc"
      "${ENGINE_SRC_DIR}/online/psm_clg_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/noise_filt/noise_filter_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/itn/text_normalizer_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/lu/lu_processor_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/punctuation/predictor_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/punctuation/albert_punctuator_manager_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/punctuation/albert_punctuator_test.cc"
      "${ENGINE_SRC_DIR}/post_processor/semantic/semantic_processor_test.cc"
      "${ENGINE_SRC_DIR}/util/engine_util_test.cc"
      "${ENGINE_SRC_DIR}/util/bert_tokenizer_test.cc"
      "${ENGINE_SRC_DIR}/util/asr_tokenizer_test.cc"
      "${ENGINE_SRC_DIR}/util/pinyin_segmenter_test.cc"
      "${ENGINE_SRC_DIR}/vad/snr_voice_detector_test.cc"
      "${GRAMMAR_SRC_DIR}/psm_pool_test.cc"
      "${GRAMMAR_SRC_DIR}/chinese_converter_test.cc"
      "${GRAMMAR_SRC_DIR}/parser/grammar_parser_test.cc"
    )
    list(REMOVE_ITEM test_src_simple
      ${test_src_exclude}
    )
    set(testdata "./testdata")

    foreach (sourcefile ${test_src_simple})
      string(REPLACE "${ENGINE_SRC_DIR}/" "" exename ${sourcefile})
      string(REPLACE "${GRAMMAR_SRC_DIR}/" "" exename ${exename})
      string(REPLACE ".cc" "" exename ${exename})
      string(REPLACE "/" "_" exename ${exename})
      add_executable(${exename}
        ${sourcefile}
      )

      target_link_libraries(${exename} gmock gtest dl mobvoi_recognizer)
      if ((${OS} STREQUAL "embedded_linux") AND (${ARCH} STREQUAL "x86_64"))
        target_link_libraries(${exename} gcov)
        target_compile_options(${exename} PRIVATE -fprofile-arcs)
        target_compile_options(${exename} PRIVATE -ftest-coverage)
      endif()
    endforeach ()
  endif()
endif ()

if (${OS} STREQUAL "android")
  add_executable(dnn_decodable_test
    ${ENGINE_SRC_DIR}/dnn/dnn_decodable_util.cc
    ${ENGINE_SRC_DIR}/dnn/dnn_decodable_test.cc
  )
  target_link_libraries(dnn_decodable_test mobvoi_recognizer gtest)
endif ()

if (${OS} STREQUAL "linux")
  # won't reach here for now
  add_executable(recognizer_server_main ${SERVER_SRC})
  target_link_libraries(recognizer_server_main dl mobvoi_recognizer)
endif ()
