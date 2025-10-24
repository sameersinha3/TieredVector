cmake_minimum_required(VERSION 3.16)
project(TieredVector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find RocksDB
find_package(PkgConfig REQUIRED)
pkg_check_modules(ROCKSDB REQUIRED rocksdb)

# Find hiredis (Redis C client)
find_package(PkgConfig REQUIRED)
pkg_check_modules(HIREDIS REQUIRED hiredis)

# Find Google Cloud Storage (optional)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GCS google-cloud-cpp-storage)

# Include directories
include_directories(${ROCKSDB_INCLUDE_DIRS})
include_directories(${HIREDIS_INCLUDE_DIRS})
include_directories(src)

# Add GCS includes if available
if(GCS_FOUND)
    include_directories(${GCS_INCLUDE_DIRS})
    add_definitions(-DHAVE_GCS)
endif()

# Add executable
add_executable(tiered_vector 
    src/main.cpp
    src/storage_manager.cpp
    src/http_server.cpp
)

# Link libraries
target_link_libraries(tiered_vector 
    ${ROCKSDB_LIBRARIES}
    ${HIREDIS_LIBRARIES}
)

# Add GCS libraries if available
if(GCS_FOUND)
    target_link_libraries(tiered_vector ${GCS_LIBRARIES})
endif()

# Compiler flags
target_compile_options(tiered_vector PRIVATE 
    ${ROCKSDB_CFLAGS_OTHER}
    ${HIREDIS_CFLAGS_OTHER}
)

# Add GCS flags if available
if(GCS_FOUND)
    target_compile_options(tiered_vector PRIVATE ${GCS_CFLAGS_OTHER})
endif()
