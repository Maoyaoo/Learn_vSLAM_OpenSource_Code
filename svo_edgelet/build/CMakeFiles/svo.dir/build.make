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
CMAKE_SOURCE_DIR = /home/my/Workspace/svo_edgelet/svo_edgelet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/my/Workspace/svo_edgelet/svo_edgelet/build

# Include any dependencies generated for this target.
include CMakeFiles/svo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/svo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svo.dir/flags.make

CMakeFiles/svo.dir/src/frame_handler_mono.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/frame_handler_mono.cpp.o: ../src/frame_handler_mono.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/svo.dir/src/frame_handler_mono.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/frame_handler_mono.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame_handler_mono.cpp

CMakeFiles/svo.dir/src/frame_handler_mono.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/frame_handler_mono.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame_handler_mono.cpp > CMakeFiles/svo.dir/src/frame_handler_mono.cpp.i

CMakeFiles/svo.dir/src/frame_handler_mono.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/frame_handler_mono.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame_handler_mono.cpp -o CMakeFiles/svo.dir/src/frame_handler_mono.cpp.s

CMakeFiles/svo.dir/src/frame_handler_base.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/frame_handler_base.cpp.o: ../src/frame_handler_base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/svo.dir/src/frame_handler_base.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/frame_handler_base.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame_handler_base.cpp

CMakeFiles/svo.dir/src/frame_handler_base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/frame_handler_base.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame_handler_base.cpp > CMakeFiles/svo.dir/src/frame_handler_base.cpp.i

CMakeFiles/svo.dir/src/frame_handler_base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/frame_handler_base.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame_handler_base.cpp -o CMakeFiles/svo.dir/src/frame_handler_base.cpp.s

CMakeFiles/svo.dir/src/frame.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/frame.cpp.o: ../src/frame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/svo.dir/src/frame.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/frame.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame.cpp

CMakeFiles/svo.dir/src/frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/frame.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame.cpp > CMakeFiles/svo.dir/src/frame.cpp.i

CMakeFiles/svo.dir/src/frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/frame.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/frame.cpp -o CMakeFiles/svo.dir/src/frame.cpp.s

CMakeFiles/svo.dir/src/point.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/point.cpp.o: ../src/point.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/svo.dir/src/point.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/point.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/point.cpp

CMakeFiles/svo.dir/src/point.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/point.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/point.cpp > CMakeFiles/svo.dir/src/point.cpp.i

CMakeFiles/svo.dir/src/point.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/point.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/point.cpp -o CMakeFiles/svo.dir/src/point.cpp.s

CMakeFiles/svo.dir/src/map.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/map.cpp.o: ../src/map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/svo.dir/src/map.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/map.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/map.cpp

CMakeFiles/svo.dir/src/map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/map.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/map.cpp > CMakeFiles/svo.dir/src/map.cpp.i

CMakeFiles/svo.dir/src/map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/map.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/map.cpp -o CMakeFiles/svo.dir/src/map.cpp.s

CMakeFiles/svo.dir/src/pose_optimizer.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/pose_optimizer.cpp.o: ../src/pose_optimizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/svo.dir/src/pose_optimizer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/pose_optimizer.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/pose_optimizer.cpp

CMakeFiles/svo.dir/src/pose_optimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/pose_optimizer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/pose_optimizer.cpp > CMakeFiles/svo.dir/src/pose_optimizer.cpp.i

CMakeFiles/svo.dir/src/pose_optimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/pose_optimizer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/pose_optimizer.cpp -o CMakeFiles/svo.dir/src/pose_optimizer.cpp.s

CMakeFiles/svo.dir/src/initialization.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/initialization.cpp.o: ../src/initialization.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/svo.dir/src/initialization.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/initialization.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/initialization.cpp

CMakeFiles/svo.dir/src/initialization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/initialization.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/initialization.cpp > CMakeFiles/svo.dir/src/initialization.cpp.i

CMakeFiles/svo.dir/src/initialization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/initialization.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/initialization.cpp -o CMakeFiles/svo.dir/src/initialization.cpp.s

CMakeFiles/svo.dir/src/matcher.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/matcher.cpp.o: ../src/matcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/svo.dir/src/matcher.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/matcher.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/matcher.cpp

CMakeFiles/svo.dir/src/matcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/matcher.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/matcher.cpp > CMakeFiles/svo.dir/src/matcher.cpp.i

CMakeFiles/svo.dir/src/matcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/matcher.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/matcher.cpp -o CMakeFiles/svo.dir/src/matcher.cpp.s

CMakeFiles/svo.dir/src/reprojector.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/reprojector.cpp.o: ../src/reprojector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/svo.dir/src/reprojector.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/reprojector.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/reprojector.cpp

CMakeFiles/svo.dir/src/reprojector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/reprojector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/reprojector.cpp > CMakeFiles/svo.dir/src/reprojector.cpp.i

CMakeFiles/svo.dir/src/reprojector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/reprojector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/reprojector.cpp -o CMakeFiles/svo.dir/src/reprojector.cpp.s

CMakeFiles/svo.dir/src/feature_alignment.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/feature_alignment.cpp.o: ../src/feature_alignment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/svo.dir/src/feature_alignment.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/feature_alignment.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/feature_alignment.cpp

CMakeFiles/svo.dir/src/feature_alignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/feature_alignment.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/feature_alignment.cpp > CMakeFiles/svo.dir/src/feature_alignment.cpp.i

CMakeFiles/svo.dir/src/feature_alignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/feature_alignment.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/feature_alignment.cpp -o CMakeFiles/svo.dir/src/feature_alignment.cpp.s

CMakeFiles/svo.dir/src/feature_detection.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/feature_detection.cpp.o: ../src/feature_detection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/svo.dir/src/feature_detection.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/feature_detection.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/feature_detection.cpp

CMakeFiles/svo.dir/src/feature_detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/feature_detection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/feature_detection.cpp > CMakeFiles/svo.dir/src/feature_detection.cpp.i

CMakeFiles/svo.dir/src/feature_detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/feature_detection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/feature_detection.cpp -o CMakeFiles/svo.dir/src/feature_detection.cpp.s

CMakeFiles/svo.dir/src/depth_filter.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/depth_filter.cpp.o: ../src/depth_filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/svo.dir/src/depth_filter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/depth_filter.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/depth_filter.cpp

CMakeFiles/svo.dir/src/depth_filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/depth_filter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/depth_filter.cpp > CMakeFiles/svo.dir/src/depth_filter.cpp.i

CMakeFiles/svo.dir/src/depth_filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/depth_filter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/depth_filter.cpp -o CMakeFiles/svo.dir/src/depth_filter.cpp.s

CMakeFiles/svo.dir/src/config.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/config.cpp.o: ../src/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/svo.dir/src/config.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/config.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/config.cpp

CMakeFiles/svo.dir/src/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/config.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/config.cpp > CMakeFiles/svo.dir/src/config.cpp.i

CMakeFiles/svo.dir/src/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/config.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/config.cpp -o CMakeFiles/svo.dir/src/config.cpp.s

CMakeFiles/svo.dir/src/camera_model.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/camera_model.cpp.o: ../src/camera_model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/svo.dir/src/camera_model.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/camera_model.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/camera_model.cpp

CMakeFiles/svo.dir/src/camera_model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/camera_model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/camera_model.cpp > CMakeFiles/svo.dir/src/camera_model.cpp.i

CMakeFiles/svo.dir/src/camera_model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/camera_model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/camera_model.cpp -o CMakeFiles/svo.dir/src/camera_model.cpp.s

CMakeFiles/svo.dir/src/sparse_align.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/sparse_align.cpp.o: ../src/sparse_align.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/svo.dir/src/sparse_align.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/sparse_align.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/sparse_align.cpp

CMakeFiles/svo.dir/src/sparse_align.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/sparse_align.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/sparse_align.cpp > CMakeFiles/svo.dir/src/sparse_align.cpp.i

CMakeFiles/svo.dir/src/sparse_align.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/sparse_align.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/sparse_align.cpp -o CMakeFiles/svo.dir/src/sparse_align.cpp.s

CMakeFiles/svo.dir/src/debug.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/debug.cpp.o: ../src/debug.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/svo.dir/src/debug.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/debug.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/debug.cpp

CMakeFiles/svo.dir/src/debug.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/debug.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/debug.cpp > CMakeFiles/svo.dir/src/debug.cpp.i

CMakeFiles/svo.dir/src/debug.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/debug.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/debug.cpp -o CMakeFiles/svo.dir/src/debug.cpp.s

CMakeFiles/svo.dir/src/math_utils.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/math_utils.cpp.o: ../src/math_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/svo.dir/src/math_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/math_utils.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/math_utils.cpp

CMakeFiles/svo.dir/src/math_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/math_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/math_utils.cpp > CMakeFiles/svo.dir/src/math_utils.cpp.i

CMakeFiles/svo.dir/src/math_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/math_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/math_utils.cpp -o CMakeFiles/svo.dir/src/math_utils.cpp.s

CMakeFiles/svo.dir/src/homography.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/homography.cpp.o: ../src/homography.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object CMakeFiles/svo.dir/src/homography.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/homography.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/homography.cpp

CMakeFiles/svo.dir/src/homography.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/homography.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/homography.cpp > CMakeFiles/svo.dir/src/homography.cpp.i

CMakeFiles/svo.dir/src/homography.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/homography.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/homography.cpp -o CMakeFiles/svo.dir/src/homography.cpp.s

CMakeFiles/svo.dir/src/robust_cost.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/robust_cost.cpp.o: ../src/robust_cost.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building CXX object CMakeFiles/svo.dir/src/robust_cost.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/robust_cost.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/robust_cost.cpp

CMakeFiles/svo.dir/src/robust_cost.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/robust_cost.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/robust_cost.cpp > CMakeFiles/svo.dir/src/robust_cost.cpp.i

CMakeFiles/svo.dir/src/robust_cost.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/robust_cost.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/robust_cost.cpp -o CMakeFiles/svo.dir/src/robust_cost.cpp.s

CMakeFiles/svo.dir/src/fast_10_score.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/fast_10_score.cpp.o: ../src/fast_10_score.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Building CXX object CMakeFiles/svo.dir/src/fast_10_score.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/fast_10_score.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_10_score.cpp

CMakeFiles/svo.dir/src/fast_10_score.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/fast_10_score.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_10_score.cpp > CMakeFiles/svo.dir/src/fast_10_score.cpp.i

CMakeFiles/svo.dir/src/fast_10_score.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/fast_10_score.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_10_score.cpp -o CMakeFiles/svo.dir/src/fast_10_score.cpp.s

CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.o: ../src/fast_nonmax_3x3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_21) "Building CXX object CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_nonmax_3x3.cpp

CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_nonmax_3x3.cpp > CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.i

CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_nonmax_3x3.cpp -o CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.s

CMakeFiles/svo.dir/src/fast_10.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/fast_10.cpp.o: ../src/fast_10.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_22) "Building CXX object CMakeFiles/svo.dir/src/fast_10.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/fast_10.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_10.cpp

CMakeFiles/svo.dir/src/fast_10.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/fast_10.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_10.cpp > CMakeFiles/svo.dir/src/fast_10.cpp.i

CMakeFiles/svo.dir/src/fast_10.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/fast_10.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/fast_10.cpp -o CMakeFiles/svo.dir/src/fast_10.cpp.s

CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.o: ../src/faster_corner_10_sse.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_23) "Building CXX object CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/faster_corner_10_sse.cpp

CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/faster_corner_10_sse.cpp > CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.i

CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/faster_corner_10_sse.cpp -o CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.s

CMakeFiles/svo.dir/src/slamviewer.cpp.o: CMakeFiles/svo.dir/flags.make
CMakeFiles/svo.dir/src/slamviewer.cpp.o: ../src/slamviewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_24) "Building CXX object CMakeFiles/svo.dir/src/slamviewer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svo.dir/src/slamviewer.cpp.o -c /home/my/Workspace/svo_edgelet/svo_edgelet/src/slamviewer.cpp

CMakeFiles/svo.dir/src/slamviewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svo.dir/src/slamviewer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/my/Workspace/svo_edgelet/svo_edgelet/src/slamviewer.cpp > CMakeFiles/svo.dir/src/slamviewer.cpp.i

CMakeFiles/svo.dir/src/slamviewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svo.dir/src/slamviewer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/my/Workspace/svo_edgelet/svo_edgelet/src/slamviewer.cpp -o CMakeFiles/svo.dir/src/slamviewer.cpp.s

# Object files for target svo
svo_OBJECTS = \
"CMakeFiles/svo.dir/src/frame_handler_mono.cpp.o" \
"CMakeFiles/svo.dir/src/frame_handler_base.cpp.o" \
"CMakeFiles/svo.dir/src/frame.cpp.o" \
"CMakeFiles/svo.dir/src/point.cpp.o" \
"CMakeFiles/svo.dir/src/map.cpp.o" \
"CMakeFiles/svo.dir/src/pose_optimizer.cpp.o" \
"CMakeFiles/svo.dir/src/initialization.cpp.o" \
"CMakeFiles/svo.dir/src/matcher.cpp.o" \
"CMakeFiles/svo.dir/src/reprojector.cpp.o" \
"CMakeFiles/svo.dir/src/feature_alignment.cpp.o" \
"CMakeFiles/svo.dir/src/feature_detection.cpp.o" \
"CMakeFiles/svo.dir/src/depth_filter.cpp.o" \
"CMakeFiles/svo.dir/src/config.cpp.o" \
"CMakeFiles/svo.dir/src/camera_model.cpp.o" \
"CMakeFiles/svo.dir/src/sparse_align.cpp.o" \
"CMakeFiles/svo.dir/src/debug.cpp.o" \
"CMakeFiles/svo.dir/src/math_utils.cpp.o" \
"CMakeFiles/svo.dir/src/homography.cpp.o" \
"CMakeFiles/svo.dir/src/robust_cost.cpp.o" \
"CMakeFiles/svo.dir/src/fast_10_score.cpp.o" \
"CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.o" \
"CMakeFiles/svo.dir/src/fast_10.cpp.o" \
"CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.o" \
"CMakeFiles/svo.dir/src/slamviewer.cpp.o"

# External object files for target svo
svo_EXTERNAL_OBJECTS =

../lib/libsvo.so: CMakeFiles/svo.dir/src/frame_handler_mono.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/frame_handler_base.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/frame.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/point.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/map.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/pose_optimizer.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/initialization.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/matcher.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/reprojector.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/feature_alignment.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/feature_detection.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/depth_filter.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/config.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/camera_model.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/sparse_align.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/debug.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/math_utils.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/homography.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/robust_cost.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/fast_10_score.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/fast_nonmax_3x3.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/fast_10.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/faster_corner_10_sse.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/src/slamviewer.cpp.o
../lib/libsvo.so: CMakeFiles/svo.dir/build.make
../lib/libsvo.so: /usr/local/lib/libopencv_dnn.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_highgui.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_ml.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_objdetect.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_shape.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_stitching.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_superres.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_videostab.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_viz.so.3.4.15
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
../lib/libsvo.so: /usr/local/lib/libpangolin.so
../lib/libsvo.so: /usr/local/lib/libopencv_calib3d.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_features2d.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_flann.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_photo.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_video.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_videoio.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_imgproc.so.3.4.15
../lib/libsvo.so: /usr/local/lib/libopencv_core.so.3.4.15
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.71.0
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libGLX.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libEGL.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libSM.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libICE.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libX11.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libXext.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libGLX.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libEGL.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libSM.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libICE.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libX11.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libXext.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libdc1394.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libavcodec.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libavformat.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libavutil.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libswscale.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libavdevice.so
../lib/libsvo.so: /usr/lib/libOpenNI.so
../lib/libsvo.so: /usr/lib/libOpenNI2.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libpng.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libz.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libtiff.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/libzstd.so
../lib/libsvo.so: /usr/lib/x86_64-linux-gnu/liblz4.so
../lib/libsvo.so: CMakeFiles/svo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_25) "Linking CXX shared library ../lib/libsvo.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/svo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svo.dir/build: ../lib/libsvo.so

.PHONY : CMakeFiles/svo.dir/build

CMakeFiles/svo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/svo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/svo.dir/clean

CMakeFiles/svo.dir/depend:
	cd /home/my/Workspace/svo_edgelet/svo_edgelet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/my/Workspace/svo_edgelet/svo_edgelet /home/my/Workspace/svo_edgelet/svo_edgelet /home/my/Workspace/svo_edgelet/svo_edgelet/build /home/my/Workspace/svo_edgelet/svo_edgelet/build /home/my/Workspace/svo_edgelet/svo_edgelet/build/CMakeFiles/svo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svo.dir/depend
