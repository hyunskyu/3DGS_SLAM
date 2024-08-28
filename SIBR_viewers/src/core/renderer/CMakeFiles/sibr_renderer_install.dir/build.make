# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jskim/gaussian-splatting/SIBR_viewers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jskim/gaussian-splatting/SIBR_viewers

# Utility rule file for sibr_renderer_install.

# Include any custom commands dependencies for this target.
include src/core/renderer/CMakeFiles/sibr_renderer_install.dir/compiler_depend.make

# Include the progress variables for this target.
include src/core/renderer/CMakeFiles/sibr_renderer_install.dir/progress.make

src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/AddShadowRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/AddShadowRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/BinaryMeshRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/BinaryMeshRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/BlurRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/BlurRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/ColoredMeshRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/ColoredMeshRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/Config.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/CopyRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/CopyRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/DepthRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/DepthRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/NormalRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/NormalRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/PointBasedRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/PointBasedRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/PoissonRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/PoissonRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/PositionRender.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/PositionRender.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/RenderMaskHolder.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/RenderMaskHolder.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/ShadowMapRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/ShadowMapRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/TexturedMeshRenderer.cpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/TexturedMeshRenderer.hpp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/addshadow.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/blur.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/colored_mesh.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/colored_mesh.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/copy.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/copy_depth.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/depthRenderer.fp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/depthRenderer.vp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/emotive_relight.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/emotive_relight.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/hdrEnvMap.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/hdrEnvMap.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/longlat.gp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/longlat.vp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/longlatColor.fp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/longlatDepth.fp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/noproj.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/normalRenderer.fp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/normalRenderer.vp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/normalRendererGen.gp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/normalRendererGen.vp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/poisson_diverg.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/poisson_interp.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/poisson_jacobi.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/poisson_restrict.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/positionReflectedDirRenderer.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/positionReflectedDirRenderer.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/positionRenderer.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/positionRenderer.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/shadowMapRenderer.fp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/shadowMapRenderer.vp
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/texture-invert.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/texture.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/texture.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/textured_mesh.frag
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/textured_mesh.vert
src/core/renderer/CMakeFiles/sibr_renderer_install: src/core/renderer/shaders/textured_mesh_flipY.vert
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jskim/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "run the installation only for sibr_renderer"
	cd /home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer && /usr/bin/cmake -DBUILD_TYPE=Release -DCOMPONENT=sibr_renderer_install -P /home/jskim/gaussian-splatting/SIBR_viewers/cmake_install.cmake

sibr_renderer_install: src/core/renderer/CMakeFiles/sibr_renderer_install
sibr_renderer_install: src/core/renderer/CMakeFiles/sibr_renderer_install.dir/build.make
.PHONY : sibr_renderer_install

# Rule to build all files generated by this target.
src/core/renderer/CMakeFiles/sibr_renderer_install.dir/build: sibr_renderer_install
.PHONY : src/core/renderer/CMakeFiles/sibr_renderer_install.dir/build

src/core/renderer/CMakeFiles/sibr_renderer_install.dir/clean:
	cd /home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer && $(CMAKE_COMMAND) -P CMakeFiles/sibr_renderer_install.dir/cmake_clean.cmake
.PHONY : src/core/renderer/CMakeFiles/sibr_renderer_install.dir/clean

src/core/renderer/CMakeFiles/sibr_renderer_install.dir/depend:
	cd /home/jskim/gaussian-splatting/SIBR_viewers && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jskim/gaussian-splatting/SIBR_viewers /home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer /home/jskim/gaussian-splatting/SIBR_viewers /home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer /home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/CMakeFiles/sibr_renderer_install.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/core/renderer/CMakeFiles/sibr_renderer_install.dir/depend

