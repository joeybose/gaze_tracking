#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
cmake_minimum_required(VERSION 2.8.4)

PROJECT(gaze-tracking)
include(dlib/cmake)
#set(CMAKE_MODULE_PATH "/usr/local/Cellar/opencv/2.4.12/share/OpenCV" ${CMAKE_MODULE_PATH})
#set(OpenCV_DIR "/usr/local/Cellar/opencv/2.4.12/")

file(COPY src/models DESTINATION ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(FRAMEWORK_LINKER "-framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo")

# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
MACRO(add_example name)
   # Commented out until Features.cpp compiles.
   #ADD_EXECUTABLE(${name} src/${name}.cpp src/Features.h src/Features.cpp)
   ADD_EXECUTABLE(${name} src/${name}.cpp)
   TARGET_LINK_LIBRARIES(${name} dlib ${OpenCV_LIBS} ${GL} ${GLFW_STATIC_LIBRARIES} ${PYTHON_LIBRARIES} ${FRAMEWORK_LINKER})
   message(${GL})
ENDMACRO()

#find_package(OpenCV24 REQUIRED)
# find_package(OpenCV 2.4 EXACT REQUIRED)
find_package(OpenCV 2.4.12 EXACT REQUIRED
	HINTS "/usr/local/cellar/opencv/2.4.12/"
)

if (OpenCV_FOUND)
   find_package(PkgConfig REQUIRED)
   pkg_search_module(GLFW REQUIRED glfw3)
   find_library(GL OpenGL REQUIRED)
   include_directories(${OpenCV_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

# Find Python
    include(FindPackageHandleStandardArgs)

    if (PYTHON_EXECUTABLE AND EXISTS ${PYTHON_EXECUTABLE}-config)
        set(PYTHON_CONFIG ${PYTHON_EXECUTABLE}-config CACHE PATH "" FORCE)
    else ()
        find_program(PYTHON_CONFIG
            NAMES python-config python-config2.7 python-config2.6 python-config2.6
                  python-config2.4 python-config2.3)
    endif ()

    # The OpenBSD python packages have python-config's that don't reliably
    # report linking flags that will work.
    if (PYTHON_CONFIG AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "OpenBSD")
        execute_process(COMMAND "${PYTHON_CONFIG}" --ldflags
                        OUTPUT_VARIABLE PYTHON_LIBRARIES
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        execute_process(COMMAND "${PYTHON_CONFIG}" --includes
                        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)

        string(REGEX REPLACE "^[-I]" "" PYTHON_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
        string(REGEX REPLACE "[ ]-I" " " PYTHON_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
        separate_arguments(PYTHON_INCLUDE_DIR)

        find_package_handle_standard_args(PythonDev DEFAULT_MSG
            PYTHON_CONFIG
            PYTHON_INCLUDE_DIR
            PYTHON_LIBRARIES
        )
    else ()
        find_package(PythonLibs)
        if (PYTHON_INCLUDE_PATH AND NOT PYTHON_INCLUDE_DIR)
            set(PYTHON_INCLUDE_DIR "${PYTHON_INCLUDE_PATH}")
        endif ()
        find_package_handle_standard_args(PythonDev DEFAULT_MSG
            PYTHON_INCLUDE_DIR
            PYTHON_LIBRARIES
        )
    endif ()

# Find NumPy
    find_package(PythonInterp REQUIRED)

    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "import numpy as n; print(n.__version__); print(n.get_include());"
        RESULT_VARIABLE _NUMPY_SEARCH_SUCCESS
        OUTPUT_VARIABLE _NUMPY_VALUES_OUTPUT
        ERROR_VARIABLE _NUMPY_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT _NUMPY_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR
            "NumPy import failure:\n${_NUMPY_ERROR_VALUE}")
        return()
    endif()

    # Convert the process output into a list
    string(REGEX REPLACE ";" "\\\\;" _NUMPY_VALUES ${_NUMPY_VALUES_OUTPUT})
    string(REGEX REPLACE "\n" ";" _NUMPY_VALUES ${_NUMPY_VALUES})
    # Just in case there is unexpected output from the Python command.
    list(GET _NUMPY_VALUES -2 NUMPY_VERSION)
    list(GET _NUMPY_VALUES -1 NUMPY_INCLUDE_DIRS)

    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _VER_CHECK "${NUMPY_VERSION}")
    if("${_VER_CHECK}" STREQUAL "")
        # The output from Python was unexpected. Raise an error always
        # here, because we found NumPy, but it appears to be corrupted somehow.
        message(FATAL_ERROR
            "Requested version and include path from NumPy, got instead:\n${_NUMPY_VALUES_OUTPUT}\n")
        return()
    endif()

    # Make sure all directory separators are '/'
    string(REGEX REPLACE "\\\\" "/" NUMPY_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS})

    # Get the major and minor version numbers
    string(REGEX REPLACE "\\." ";" _NUMPY_VERSION_LIST ${NUMPY_VERSION})
    list(GET _NUMPY_VERSION_LIST 0 NUMPY_VERSION_MAJOR)
    list(GET _NUMPY_VERSION_LIST 1 NUMPY_VERSION_MINOR)
    list(GET _NUMPY_VERSION_LIST 2 NUMPY_VERSION_PATCH)
    string(REGEX MATCH "[0-9]*" NUMPY_VERSION_PATCH ${NUMPY_VERSION_PATCH})
    math(EXPR NUMPY_VERSION_DECIMAL
        "(${NUMPY_VERSION_MAJOR} * 10000) + (${NUMPY_VERSION_MINOR} * 100) + ${NUMPY_VERSION_PATCH}")

   include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})

   add_example(ml_gaze_tracker_1)
   add_example(gaze_matrix)
   #add_example(gaze_tracker_1)
   #add_example(history_gaze_tracker)
endif()
