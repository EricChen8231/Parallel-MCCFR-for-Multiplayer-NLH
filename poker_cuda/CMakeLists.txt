cmake_minimum_required(VERSION 3.18)
project(PokerCUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ---------------------------------------------------------------------------
# CUDA architectures
#   70 = V100 (CARC Discovery)
#   80 = A100 (CARC Discovery)
#   86 = RTX 30xx (local dev)
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES "70;80;86" CACHE STRING "CUDA compute capabilities")

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
set(CMAKE_CUDA_FLAGS         "${CMAKE_CUDA_FLAGS} -lineinfo")   # for Nsight

find_package(CUDAToolkit REQUIRED)

# Optional MPI (for multi-node on CARC)
find_package(MPI)

# Optional NCCL (for multi-GPU AllReduce — required for multi-GPU training)
find_library(NCCL_LIB nccl HINTS /usr/lib /usr/local/lib $ENV{NCCL_HOME}/lib)
find_path(NCCL_INCLUDE nccl.h HINTS /usr/include /usr/local/include $ENV{NCCL_HOME}/include)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(poker_cuda
    src/main.cpp
    src/hand_eval.cpp
    src/abstraction.cpp
    src/strategy.cpp
    src/eval.cpp
    src/bot.cpp
    src/cfr_gpu.cu
)

target_include_directories(poker_cuda PRIVATE src)

target_link_libraries(poker_cuda
    CUDA::curand
    CUDA::cudart
)

if(MPI_FOUND)
    message(STATUS "MPI found — enabling multi-node support")
    target_compile_definitions(poker_cuda PRIVATE USE_MPI)
    target_link_libraries(poker_cuda MPI::MPI_CXX)
else()
    message(STATUS "MPI not found — single-node only")
endif()

if(NCCL_LIB AND NCCL_INCLUDE)
    message(STATUS "NCCL found — enabling multi-GPU AllReduce: ${NCCL_LIB}")
    target_compile_definitions(poker_cuda PRIVATE USE_NCCL)
    target_include_directories(poker_cuda PRIVATE ${NCCL_INCLUDE})
    target_link_libraries(poker_cuda ${NCCL_LIB})
else()
    message(STATUS "NCCL not found — single-GPU only (compile with -DNCCL_HOME=... to enable)")
endif()

# ---------------------------------------------------------------------------
# Separate tool: generate Two Plus Two handranks.dat
# ---------------------------------------------------------------------------
add_executable(gen_table tools/gen_lookup_table.cpp)
