# Install script for directory: /home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jskim/gaussian-splatting/SIBR_viewers/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/addshadow.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/blur.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/colored_mesh.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/colored_mesh.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/copy.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/copy_depth.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/depthRenderer.fp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/depthRenderer.vp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/emotive_relight.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/emotive_relight.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/hdrEnvMap.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/hdrEnvMap.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/longlat.gp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/longlat.vp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/longlatColor.fp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/longlatDepth.fp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/noproj.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/normalRenderer.fp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/normalRenderer.vp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/normalRendererGen.gp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/normalRendererGen.vp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/poisson_diverg.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/poisson_interp.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/poisson_jacobi.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/poisson_restrict.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/positionReflectedDirRenderer.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/positionReflectedDirRenderer.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/positionRenderer.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/positionRenderer.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/shadowMapRenderer.fp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/shadowMapRenderer.vp;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/texture-invert.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/texture.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/texture.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/textured_mesh.frag;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/textured_mesh.vert;/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core/textured_mesh_flipY.vert")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/jskim/gaussian-splatting/SIBR_viewers/install/shaders/core" TYPE PROGRAM FILES
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/addshadow.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/blur.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/colored_mesh.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/colored_mesh.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/copy.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/copy_depth.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/depthRenderer.fp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/depthRenderer.vp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/emotive_relight.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/emotive_relight.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/hdrEnvMap.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/hdrEnvMap.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/longlat.gp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/longlat.vp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/longlatColor.fp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/longlatDepth.fp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/noproj.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/normalRenderer.fp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/normalRenderer.vp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/normalRendererGen.gp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/normalRendererGen.vp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/poisson_diverg.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/poisson_interp.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/poisson_jacobi.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/poisson_restrict.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/positionReflectedDirRenderer.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/positionReflectedDirRenderer.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/positionRenderer.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/positionRenderer.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/shadowMapRenderer.fp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/shadowMapRenderer.vp"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/texture-invert.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/texture.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/texture.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/textured_mesh.frag"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/textured_mesh.vert"
      "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/shaders/textured_mesh_flipY.vert"
      )
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xsibr_renderer_installx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/jskim/gaussian-splatting/SIBR_viewers/install/lib/libsibr_renderer.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/jskim/gaussian-splatting/SIBR_viewers/install/lib" TYPE SHARED_LIBRARY FILES "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/libsibr_renderer.so")
    if(EXISTS "$ENV{DESTDIR}/home/jskim/gaussian-splatting/SIBR_viewers/install/lib/libsibr_renderer.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/jskim/gaussian-splatting/SIBR_viewers/install/lib/libsibr_renderer.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/jskim/gaussian-splatting/SIBR_viewers/install/lib/libsibr_renderer.so")
      endif()
    endif()
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xsibr_renderer_installx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xsibr_renderer_installx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/jskim/gaussian-splatting/SIBR_viewers/install/bin/libsibr_renderer.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/jskim/gaussian-splatting/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "/home/jskim/gaussian-splatting/SIBR_viewers/src/core/renderer/libsibr_renderer.so")
    if(EXISTS "$ENV{DESTDIR}/home/jskim/gaussian-splatting/SIBR_viewers/install/bin/libsibr_renderer.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/jskim/gaussian-splatting/SIBR_viewers/install/bin/libsibr_renderer.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/jskim/gaussian-splatting/SIBR_viewers/install/bin/libsibr_renderer.so")
      endif()
    endif()
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xsibr_renderer_installx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
endif()

