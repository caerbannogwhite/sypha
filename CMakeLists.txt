# CMakeList.txt: progetto CMake per Sypha. Includere l'origine e definire
# qui la logica specifica del progetto.
#
cmake_minimum_required (VERSION 3.8)

project ("Sypha")

# Aggiungere l'origine all'eseguibile di questo progetto.
add_executable (Sypha
	"src/sypha.cpp" "src/sypha.h"
	"src/common.cpp" "src/common.h"
)

# VCPGK bin added to PATH
set(VCPKG_INSTALLED_PATH "C:\\Users\\IP 520S-14IKB 96IX\\coding\\vcpkg\\installed\\x64-windows")

# To find and use boost-program_options header
find_path(BOOST_PROGRAM_OPTIONS_INCLUDE_DIR boost/program_options.hpp)
include_directories(${BOOST_PROGRAM_OPTIONS_INCLUDE_DIR})

target_link_libraries(Sypha "${VCPKG_INSTALLED_PATH}\\lib\\boost_program_options-vc140-mt.lib")
