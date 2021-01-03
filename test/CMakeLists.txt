set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(mono_se2 ${PROJECT_SOURCE_DIR}/Examples/Monocular/mono_se2.cc)
target_link_libraries(mono_se2 ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(test_mono_se2_rk_dataset test_mono_se2_rk_dataset.cc)
target_link_libraries(test_mono_se2_rk_dataset ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(test_orbDetector test_orbDetector.cpp)
target_link_libraries(test_orbDetector ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(test_orbMatcher test_orbMatcher.cpp)
target_link_libraries(test_orbMatcher ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(test_featureExtract
    test_featureExtract.cpp
    gms_matcher.h)
target_link_libraries(test_featureExtract ${PROJECT_NAME})

add_executable(test_kltTracking
    test_kltTracking.cpp
    gms_matcher.h)
target_link_libraries(test_kltTracking ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(test_mono_fzu_dataset test_mono_fzu_dataset.cpp)
target_link_libraries(test_mono_fzu_dataset ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(match_odometry match_odometry.cpp)
target_link_libraries(match_odometry ${Boost_LIBRARIES} ${EIGEN3_LIBS})