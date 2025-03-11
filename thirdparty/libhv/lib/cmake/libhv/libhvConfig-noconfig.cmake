#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hv" for configuration ""
set_property(TARGET hv APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hv PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libhv.so"
  IMPORTED_SONAME_NOCONFIG "libhv.so"
  )

list(APPEND _cmake_import_check_targets hv )
list(APPEND _cmake_import_check_files_for_hv "${_IMPORT_PREFIX}/lib/libhv.so" )

# Import target "hv_static" for configuration ""
set_property(TARGET hv_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hv_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C;CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libhv_static.a"
  )

list(APPEND _cmake_import_check_targets hv_static )
list(APPEND _cmake_import_check_files_for_hv_static "${_IMPORT_PREFIX}/lib/libhv_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
