cmake_minimum_required(VERSION 3.25)
project(indexing_and_search)

set(CMAKE_CXX_STANDARD 17)

include(CGraph/cmake/CGraph-env-include.cmake)
add_link_options(-static -static-libgcc -static-libstdc++)
add_compile_options(-O3)

add_executable(index
        $<TARGET_OBJECTS:CGraph>
        index.cpp
)

add_executable(search
        $<TARGET_OBJECTS:CGraph>
        search.cpp
)