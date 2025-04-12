cmake_minimum_required(VERSION 3.10)
project(TinyNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIRD_PARTY_CMAKE_DIR "${CMAKE_SOURCE_DIR}/third_party_install/cmake")
include(${THIRD_PARTY_CMAKE_DIR}/common.cmake)
include(${THIRD_PARTY_CMAKE_DIR}/source_of_randomness.cmake)
include(${THIRD_PARTY_CMAKE_DIR}/threading.cmake)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party_install")
find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)
#set(Boost_USE_STATIC_LIBS ON) # libboost static linking
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(
    ${CMAKE_SOURCE_DIR}/third_party_install/include
    ${CMAKE_SOURCE_DIR}/third_party_install/include/SEAL-4.1
)

# Link directories
link_directories(${CMAKE_SOURCE_DIR}/third_party_install/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fconcepts")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

macro (add_exec _name)
    add_executable(${_name} "${_name}.cpp")
    target_link_libraries(${_name}  ${RELIC_LIBRARIES} ${OPENSSL_LIBRARIES} ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${TINYGARBLE_LIBRARIES} seal-4.1)
endmacro()

add_exec(PlainNN)
add_exec(TinyNN)
add_exec(SealNN)
add_exec(BenchmarkNN)

# below also works
# # Add library search path
# link_directories(${THIRD_PARTY_DIR}/lib)
#
# # Your main file
# add_executable(TinyNN TinyNN.cpp)
#
# # Link libraries from emp-tool, emp-ot etc
# target_link_libraries(TinyNN
#   ${RELIC_LIBRARIES}
#   ${OPENSSL_LIBRARIES}
#   ${Boost_LIBRARIES}
#   ${GMP_LIBRARIES}
# )
