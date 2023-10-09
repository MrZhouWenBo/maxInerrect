# 表示cmake的最低版本
cmake_minimum_required (VERSION 3.16)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR})
set(projrctName demo)


# 表示目前编译的项目
project (${projrctName})


set(opencv_INCLUDES  ${CMAKE_CURRENT_LIST_DIR}/3rd/opencv/include)
set(opencv_LIB_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/3rd/opencv/lib)
set(opencv_LIBS libopencv_world.so)



include_directories(
    ${opencv_INCLUDES}
)



link_directories(${projrctName} ${opencv_LIB_DIRECTORY} )
# 表示项目的执行程序， 括号中的day07 表示最终生成的执行程序名称，  后面的表示程序的源码文件
add_executable(${projrctName} main.cpp)

target_link_libraries(${projrctName} 
    -pthread
    ${opencv_LIBS}
 )