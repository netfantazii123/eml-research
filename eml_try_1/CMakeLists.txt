cmake_minimum_required(VERSION 3.10)
project(eml_c C)

set(CMAKE_C_STANDARD 11)

# Общий исходник
set(SRC eml_c.c)

# --- DLL (shared library) для Python ctypes ---
add_library(eml_c_dll SHARED ${SRC})
set_target_properties(eml_c_dll PROPERTIES
    OUTPUT_NAME "eml_c"
    PREFIX ""
)
# При сборке DLL не линковать main()
target_compile_definitions(eml_c_dll PRIVATE EML_BUILD_DLL)

# --- EXE (автономный запуск тестов) ---
add_executable(eml_c_exe ${SRC})
set_target_properties(eml_c_exe PROPERTIES
    OUTPUT_NAME "eml_test"
)

# Математическая библиотека (Linux/macOS)
if(UNIX)
    target_link_libraries(eml_c_dll m)
    target_link_libraries(eml_c_exe m)
endif()
