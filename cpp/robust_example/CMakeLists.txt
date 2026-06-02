file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cpp")

if(NOT TARGET robust_examples)
    add_custom_target(robust_examples)
endif()

foreach(example_src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name "${example_src}" NAME_WE)

    if(NOT TARGET "${example_name}")
        add_executable("${example_name}" "${example_src}")
        target_link_libraries("${example_name}" PRIVATE ${PROJECT_NAME})
        if(ENABLE_PROFILING)
            target_link_libraries("${example_name}" PRIVATE ${PROFILER_LIB})
        endif()
        add_dependencies(robust_examples "${example_name}")
    endif()
endforeach()
