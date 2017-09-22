cmake_minimum_required(VERSION 2.8)
PROJECT(cnn_svm)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(. ${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(~/divers_lib/caffe/include ~/divers_lib/caffe/build ~/divers_lib/caffe/build/include ~/divers_lib/caffe/build/src /usr/local/cuda/include ~/divers_lib/cudnn/include)
LINK_DIRECTORIES(/usr/local/cuda/lib64 ~/divers_lib/caffe/build/lib/ ~/divers_lib/cudnn/lib64)

ADD_EXECUTABLE(create_svm_file create_svm_file.cpp)

TARGET_LINK_LIBRARIES(create_svm_file ${OpenCV_LIBS} caffe glog cudart cudnn)
