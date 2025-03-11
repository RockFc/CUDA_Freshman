option(BUILD_TEST "Builds test party" ON)

# 添加头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/thirdparty/googletest/include
    )
# 设置库依赖
set (LIBGTEST_STATIC ${PROJECT_SOURCE_DIR}/thirdparty/googletest/lib/libgtest.a  
                    ${PROJECT_SOURCE_DIR}/thirdparty/googletest/lib/libgtest_main.a)

#设置输出文件路径
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/product/test)

if(BUILD_TEST)
    add_subdirectory(100_calculate_std)
endif()
