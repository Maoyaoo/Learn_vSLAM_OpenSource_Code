# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build

# Include any dependencies generated for this target.
include CMakeFiles/test_pipeline.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_pipeline.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_pipeline.dir/flags.make

CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.o: CMakeFiles/test_pipeline.dir/flags.make
CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.o: ../test/test_pipeline.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.o -c /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/test/test_pipeline.cpp

CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/test/test_pipeline.cpp > CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.i

CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/test/test_pipeline.cpp -o CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.s

# Object files for target test_pipeline
test_pipeline_OBJECTS = \
"CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.o"

# External object files for target test_pipeline
test_pipeline_EXTERNAL_OBJECTS =

../bin/test_pipeline: CMakeFiles/test_pipeline.dir/test/test_pipeline.cpp.o
../bin/test_pipeline: CMakeFiles/test_pipeline.dir/build.make
../bin/test_pipeline: ../lib/libsvo.so
../bin/test_pipeline: /usr/local/lib/libopencv_dnn.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_highgui.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_ml.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_objdetect.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_shape.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_stitching.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_superres.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_videostab.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_calib3d.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_features2d.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_flann.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_photo.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_video.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_videoio.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_imgproc.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_viz.so.3.4.15
../bin/test_pipeline: /usr/local/lib/libopencv_core.so.3.4.15
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.71.0
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
../bin/test_pipeline: /usr/local/lib/libpangolin.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libGLX.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libEGL.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libGLX.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libEGL.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libdc1394.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libavcodec.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libavformat.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libavutil.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libswscale.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libavdevice.so
../bin/test_pipeline: /usr/lib/libOpenNI.so
../bin/test_pipeline: /usr/lib/libOpenNI2.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libz.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/libzstd.so
../bin/test_pipeline: /usr/lib/x86_64-linux-gnu/liblz4.so
../bin/test_pipeline: CMakeFiles/test_pipeline.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test_pipeline"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_pipeline.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_pipeline.dir/build: ../bin/test_pipeline

.PHONY : CMakeFiles/test_pipeline.dir/build

CMakeFiles/test_pipeline.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_pipeline.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_pipeline.dir/clean

CMakeFiles/test_pipeline.dir/depend:
	cd /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build /home/my/Workspace/Learn_vSLAM_OpenSource_Code/svo_edgelet/build/CMakeFiles/test_pipeline.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_pipeline.dir/depend

