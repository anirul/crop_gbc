# main entry point.

add_executable(cropgbc
    main.cpp
)

target_link_libraries(cropgbc
    PUBLIC
        absl::base
        absl::flags
        absl::flags_parse
        opencv_core
        opencv_highgui 
        opencv_imgcodecs 
        opencv_imgproc
)