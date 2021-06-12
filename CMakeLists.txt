cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

## Policy
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 OLD)
endif()

## Options
SET(HAVE_G2O FALSE)     # use g2o with self built
option(WITH_LINES    "option for LINES feature"  OFF)
option(WITH_ODOMETRY "option for ODOMETRY input" OFF)
if (WITH_LINES)
    add_definitions(-DWITH_LINES=1)
    message(STATUS "Using flag: WITH_LINES")
else()
    add_definitions(-DWITH_LINES=0)
endif()
if (WITH_ODOMETRY)
    add_definitions(-DWITH_ODOMETRY=1)
    message(STATUS "Using flag: WITH_ODOMETRY")
else()
    add_definitions(-DWITH_ODOMETRY=0)
endif()

## build flags
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-deprecated-declarations -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-deprecated-declarations -Wno-unused-variable -Wno-reorder")

# DON'T FORGET to add '-march=native' to avoid segmentation fault!!!
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -ggdb -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

## packages
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
if (OpenCV_FOUND)
    message("===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    message("===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    message("===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    message("===> OpenCV Not Found!!!" )
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

## build a lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(SHARED_LIB_NAME "${PROJECT_NAME}_d")
else()
    set(SHARED_LIB_NAME "${PROJECT_NAME}")
endif()

add_library(${SHARED_LIB_NAME} SHARED
    src/System.cpp
    src/Tracking.cpp
    src/LocalMapping.cpp
    src/LoopClosing.cpp
    src/ORBextractor.cpp
    src/ORBmatcher.cpp
    src/FrameDrawer.cpp
    src/Converter.cpp
    src/MapPoint.cpp
    src/KeyFrame.cpp
    src/Atlas.cpp
    src/Map.cpp
    src/MapDrawer.cpp
    src/Optimizer.cpp
    src/PnPsolver.cpp
    src/Frame.cpp
    src/KeyFrameDatabase.cpp
    src/Sim3Solver.cpp
    src/Initializer.cpp
    src/Viewer.cpp
    src/ImuTypes.cpp
    src/OdomTypes.cpp
    src/G2oTypes.cpp
    src/CameraModels/Pinhole.cpp
    src/CameraModels/KannalaBrandt8.cpp
    src/OptimizableTypes.cpp
    src/MLPnPsolver.cpp
    src/TwoViewReconstruction.cpp
    include/System.h
    include/Tracking.h
    include/LocalMapping.h
    include/LoopClosing.h
    include/ORBextractor.h
    include/ORBmatcher.h
    include/FrameDrawer.h
    include/Converter.h
    include/MapPoint.h
    include/KeyFrame.h
    include/Atlas.h
    include/Map.h
    include/MapDrawer.h
    include/Optimizer.h
    include/PnPsolver.h
    include/Frame.h
    include/KeyFrameDatabase.h
    include/Sim3Solver.h
    include/Initializer.h
    include/Viewer.h
    include/ImuTypes.h
    include/OdomTypes.h
    include/G2oTypes.h
    include/CameraModels/GeometricCamera.h
    include/CameraModels/Pinhole.h
    include/CameraModels/KannalaBrandt8.h
    include/OptimizableTypes.h
    include/MLPnPsolver.h
    include/TwoViewReconstruction.h
)

if(${HAVE_G2O})
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
    set(G2O_ROOT /home/vance/opt/g2o-20160424/install/)
    find_package(G2O REQUIRED)
    if(G2O_FOUND)
        add_definitions(-DHAVE_G2O=1)
        include_directories(
            ${G2O_INCLUDE_DIR}
            /usr/include/suitesparse # for cholmod
        )
        list(APPEND G2O_LIBS
            ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE}
            ${G2O_SOLVER_PCG} ${G2O_TYPES_SBA} cholmod cxsparse
        )
        message("===> G2O Information: Root: ${G2O_ROOT}" )
        message("===> G2O Information: Include Dir: ${G2O_INCLUDE_DIR}" )
        message("===> G2O Information: Libs Dir: ${G2O_LIBS}" )
    else()
        message("===> Cannot find G2O! Use the 3rd files..")
        SET(HAVE_G2O FALSE)
    endif()
    target_link_libraries(${SHARED_LIB_NAME} ${G2O_LIBS})
endif()
if(NOT ${HAVE_G2O})
    add_subdirectory(Thirdparty/g2o)
    target_link_libraries(${SHARED_LIB_NAME} ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so)
endif()

target_link_libraries(
    ${SHARED_LIB_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    # ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    -lboost_serialization
    -lboost_filesystem
    -lcrypto
)

# Build examples
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Save executable files to ${PROJECT_SOURCE_DIR}/bin/debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/debug)
elseif (CMAKE_BUILD_TYPE MATCHES "Release")
message(STATUS "Save executable files to ${PROJECT_SOURCE_DIR}/bin/release")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/release)
endif()

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
# add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum ${SHARED_LIB_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
# add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${SHARED_LIB_NAME})
# add_executable(stereo_euroc Examples/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc ${SHARED_LIB_NAME})
# add_executable(stereo_tum_vi Examples/Stereo/stereo_tum_vi.cc)
# target_link_libraries(stereo_tum_vi ${SHARED_LIB_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(mono_tum Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${SHARED_LIB_NAME})
add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${SHARED_LIB_NAME})
add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${SHARED_LIB_NAME})
add_executable(mono_tum_vi Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${SHARED_LIB_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)
# add_executable(mono_inertial_euroc Examples/Monocular-Inertial/mono_inertial_euroc.cc)
# target_link_libraries(mono_inertial_euroc ${SHARED_LIB_NAME})
# add_executable(mono_inertial_tum_vi Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
# target_link_libraries(mono_inertial_tum_vi ${SHARED_LIB_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)
# add_executable(stereo_inertial_euroc Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
# target_link_libraries(stereo_inertial_euroc ${SHARED_LIB_NAME})
# add_executable(stereo_inertial_tum_vi Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
# target_link_libraries(stereo_inertial_tum_vi ${SHARED_LIB_NAME})

# Build tools
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
# add_executable(bin_vocabulary Vocabulary/bin_vocabulary.cpp)
# target_link_libraries(bin_vocabulary ${SHARED_LIB_NAME})

# Test
add_subdirectory(test)

