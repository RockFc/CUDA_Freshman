# 指定 libhv 头文件路径
set (LIBHV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/libhv/include)
# 指定 libhv 库的路径
set(LIBHV_STATIC ${PROJECT_SOURCE_DIR}/thirdparty/libhv/lib/libhv_static.a)
set(LIBHV_SHARED ${PROJECT_SOURCE_DIR}/thirdparty/libhv/lib/libhv.so)

set (CUR_SRC_DIR ${SRC_DIR}/100_calculate_std/)

# 需包含的头文件
include_directories(
    ${LIBHV_INCLUDE_DIR}
    ${CUR_SRC_DIR}
    )


add_executable(ut_std_cpu ut_std_cpu.cpp)
target_link_libraries(ut_std_cpu ${LIBGTEST_STATIC} ${LIBHV_STATIC} pthread)

add_executable(ut_std_gpu ut_std_gpu.cpp ${CUR_SRC_DIR}/std_gpu.cu )
target_link_libraries(ut_std_gpu ${LIBGTEST_STATIC} ${LIBHV_STATIC} pthread)