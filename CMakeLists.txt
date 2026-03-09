cmake_minimum_required(VERSION 3.25)
project(image-to-body-math VERSION 0.4.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
function(project_set_warnings target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
            -Wconversion -Wsign-conversion -Wnull-dereference
            -Wdouble-promotion -Wformat=2 -Wshadow
            -Wuninitialized -Wunreachable-code -Wunused
        )
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE
                -Wduplicated-cond -Wduplicated-branches
                -Wlogical-op -Wuseless-cast
            )
        endif()
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    endif()
endfunction()

# Dependencies
include(FetchContent)
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
set(FETCHCONTENT_QUIET ON)

find_package(gcem QUIET)
if(NOT gcem_FOUND)
    FetchContent_Declare(gcem GIT_REPOSITORY https://github.com/kthohr/gcem.git GIT_TAG master
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(gcem)
endif()

find_package(strong-types QUIET)
if(NOT strong-types_FOUND)
    FetchContent_Declare(strong-types GIT_REPOSITORY https://github.com/PavelGuzenfeld/strong-types.git GIT_TAG main
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(strong-types)
endif()

find_package(linalg3d QUIET)
if(NOT linalg3d_FOUND)
    FetchContent_Declare(linalg3d GIT_REPOSITORY https://github.com/PavelGuzenfeld/linalg3d.git GIT_TAG main
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(linalg3d)
endif()

find_package(fmt QUIET)
if(NOT fmt_FOUND)
    FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 11.1.4
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(fmt)
endif()

FetchContent_Declare(doctest GIT_REPOSITORY https://github.com/doctest/doctest.git GIT_TAG v2.4.11
                     SYSTEM EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(doctest)

# Header-only library
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Install public headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
)

# Export the target
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION include
)

# Install the export file
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Configure package files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Install the package configuration files
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Tests
enable_testing()

add_executable(${PROJECT_NAME}_test test/${PROJECT_NAME}_test.cpp)
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME} fmt linalg3d strong-types gcem doctest::doctest)
project_set_warnings(${PROJECT_NAME}_test)
add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)

add_executable(body_space_test test/body_space_test.cpp)
target_link_libraries(body_space_test PRIVATE ${PROJECT_NAME} fmt linalg3d strong-types gcem doctest::doctest)
project_set_warnings(body_space_test)
add_test(NAME body_space_test COMMAND body_space_test)
