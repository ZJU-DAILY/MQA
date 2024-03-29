cmake_minimum_required(VERSION 3.25)
project(indexing_and_search)

set(CMAKE_CXX_STANDARD 17)

include(CGraph/cmake/CGraph-env-include.cmake)
add_link_options(-static -static-libgcc -static-libstdc++)
add_compile_options(-O3)

add_executable(search_mr
        $<TARGET_OBJECTS:CGraph>
        search_mr.cpp)

add_executable(search_je
        $<TARGET_OBJECTS:CGraph>
        search_je.cpp)

add_executable(search_must
        $<TARGET_OBJECTS:CGraph>
        search_must.cpp
)
