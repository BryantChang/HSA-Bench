# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

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
CMAKE_SOURCE_DIR = /home/b103/Hetero-Mark-master/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/b103/Hetero-Mark-master/src

# Include any dependencies generated for this target.
include kmeans_cl12/CMakeFiles/kmeans12.dir/depend.make

# Include the progress variables for this target.
include kmeans_cl12/CMakeFiles/kmeans12.dir/progress.make

# Include the compile flags for this target's objects.
include kmeans_cl12/CMakeFiles/kmeans12.dir/flags.make

kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o: kmeans_cl12/CMakeFiles/kmeans12.dir/flags.make
kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o: kmeans_cl12/kmeans.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/b103/Hetero-Mark-master/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o"
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kmeans12.dir/kmeans.cc.o -c /home/b103/Hetero-Mark-master/src/kmeans_cl12/kmeans.cc

kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans12.dir/kmeans.cc.i"
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/b103/Hetero-Mark-master/src/kmeans_cl12/kmeans.cc > CMakeFiles/kmeans12.dir/kmeans.cc.i

kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans12.dir/kmeans.cc.s"
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/b103/Hetero-Mark-master/src/kmeans_cl12/kmeans.cc -o CMakeFiles/kmeans12.dir/kmeans.cc.s

kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.requires:
.PHONY : kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.requires

kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.provides: kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.requires
	$(MAKE) -f kmeans_cl12/CMakeFiles/kmeans12.dir/build.make kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.provides.build
.PHONY : kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.provides

kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.provides.build: kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o

# Object files for target kmeans12
kmeans12_OBJECTS = \
"CMakeFiles/kmeans12.dir/kmeans.cc.o"

# External object files for target kmeans12
kmeans12_EXTERNAL_OBJECTS =

kmeans_cl12/bin/x86_64/Release/kmeans12: kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o
kmeans_cl12/bin/x86_64/Release/kmeans12: kmeans_cl12/CMakeFiles/kmeans12.dir/build.make
kmeans_cl12/bin/x86_64/Release/kmeans12: /opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/libOpenCL.so
kmeans_cl12/bin/x86_64/Release/kmeans12: kmeans_cl12/CMakeFiles/kmeans12.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/x86_64/Release/kmeans12"
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans12.dir/link.txt --verbose=$(VERBOSE)
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && /usr/bin/cmake -E copy_if_different /home/b103/Hetero-Mark-master/src/kmeans_cl12/kmeans.cl /home/b103/Hetero-Mark-master/src/kmeans_cl12/bin/x86_64/Release/.
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && /usr/bin/cmake -E copy_if_different /home/b103/Hetero-Mark-master/src/kmeans_cl12/kmeans.cl ./

# Rule to build all files generated by this target.
kmeans_cl12/CMakeFiles/kmeans12.dir/build: kmeans_cl12/bin/x86_64/Release/kmeans12
.PHONY : kmeans_cl12/CMakeFiles/kmeans12.dir/build

kmeans_cl12/CMakeFiles/kmeans12.dir/requires: kmeans_cl12/CMakeFiles/kmeans12.dir/kmeans.cc.o.requires
.PHONY : kmeans_cl12/CMakeFiles/kmeans12.dir/requires

kmeans_cl12/CMakeFiles/kmeans12.dir/clean:
	cd /home/b103/Hetero-Mark-master/src/kmeans_cl12 && $(CMAKE_COMMAND) -P CMakeFiles/kmeans12.dir/cmake_clean.cmake
.PHONY : kmeans_cl12/CMakeFiles/kmeans12.dir/clean

kmeans_cl12/CMakeFiles/kmeans12.dir/depend:
	cd /home/b103/Hetero-Mark-master/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/b103/Hetero-Mark-master/src /home/b103/Hetero-Mark-master/src/kmeans_cl12 /home/b103/Hetero-Mark-master/src /home/b103/Hetero-Mark-master/src/kmeans_cl12 /home/b103/Hetero-Mark-master/src/kmeans_cl12/CMakeFiles/kmeans12.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : kmeans_cl12/CMakeFiles/kmeans12.dir/depend

