#############################################################################
## 
##  Copyright (C) 2019-2020 Boris Krasnopolsky, Alexey Medvedev
##  Contact: xamg-test@imec.msu.ru
## 
##  This file is part of the XAMG library.
## 
##  Commercial License Usage
##  Licensees holding valid commercial XAMG licenses may use this file in
##  accordance with the terms of commercial license agreement.
##  The license terms and conditions are subject to mutual agreement
##  between Licensee and XAMG library authors signed by both parties
##  in a written form.
## 
##  GNU General Public License Usage
##  Alternatively, this file may be used under the terms of the GNU
##  General Public License, either version 3 of the License, or (at your
##  option) any later version. The license is as published by the Free 
##  Software Foundation and appearing in the file LICENSE.GPL3 included in
##  the packaging of this file. Please review the following information to
##  ensure the GNU General Public License requirements will be met:
##  https://www.gnu.org/licenses/gpl-3.0.html.
## 
#############################################################################

#!/bin/bash

set -eu

[ -f ../env.sh ] && source ../env.sh || echo "WARNING: no environment file ../env.sh!"

BSCRIPTSDIR=../tools/dbscripts

source $BSCRIPTSDIR/base.inc
source $BSCRIPTSDIR/funcs.inc
source $BSCRIPTSDIR/compchk.inc
source $BSCRIPTSDIR/envchk.inc
source $BSCRIPTSDIR/db.inc

function dnb_hypre() {
    local pkg="hypre"
    environment_check_specific "$pkg" || fatal "${pkg}: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    any_mode_is_set "du" "$m" && du_github "hypre-space" "hypre" "v" "$V" "$m"
    local OPTS=""
    OPTS="$OPTS CC=$MPICC"
    OPTS="$OPTS CFLAGS=-O3"
    OPTS="$OPTS CXX=$MPICXX"
    OPTS="$OPTS CXXFLAGS=-O3"
    OPTS="$OPTS --with-timing"
    OPTS="$OPTS --enable-shared"
    OPTS="$OPTS --without-openmp"
    OPTS="$OPTS --without-fei"
    OPTS="$OPTS --without-mli"
    OPTS="$OPTS --disable-fortran"
    any_mode_is_set "bi" "$m" && bi_autoconf_make "$pkg" "$V" "cd src" "$OPTS" "$m"
    this_mode_is_set "i" "$m" && make_binary_symlink "$pkg" "${V}"
    return 0
}

function dnb_yaml-cpp() {
    local pkg="yaml-cpp"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    any_mode_is_set "du" "$m" && du_github "jbeder" "yaml-cpp" "yaml-cpp-" "$V" "$m"
    local OPTS=""
    OPTS="$OPTS -DYAML_BUILD_SHARED_LIBS=ON"
    OPTS="$OPTS -DYAML_CPP_BUILD_TESTS=OFF"
    OPTS="$OPTS -DYAML_CPP_BUILD_TOOLS=OFF"
    OPTS="$OPTS -DYAML_CPP_BUILD_CONTRIB=OFF"
    any_mode_is_set "bi" "$m" && bi_cmake "$pkg" "$V" ".." "$OPTS" "$m"
    this_mode_is_set "i" "$m" && make_binary_symlink "$pkg" "${V}"
    return 0
}

function dnb_argsparser() {
    local pkg="argsparser"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    any_mode_is_set "du" "$m" && du_github "a-v-medvedev" "argsparser" "v" "$V" "$m"
    if any_mode_is_set "bi" "$m"; then 
        [ -f "$INSTALL_DIR/yaml-cpp.bin/include/yaml-cpp/yaml.h" ] || fatal "$pkg: installed yaml-cpp is required to build"
    fi
    local COMMANDS=""
    local PARAMS="clean all"
    PARAMS="$PARAMS YAML_DIR=$INSTALL_DIR/yaml-cpp.bin"
    this_mode_is_set "b" "$m" && b_make "$pkg" "$V" "$COMMANDS" "$PARAMS" "$m"
    local FILES="argsparser/include/argsparser.h argsparser/libargsparser.so"
    this_mode_is_set "i" "$m" && i_direct_copy "$pkg" "$V" "$FILES" "$m"
    FILES="extensions"
    this_mode_is_set "i" "$m" && i_direct_copy "$pkg" "$V" "$FILES" "$m"
    this_mode_is_set "i" "$m" && make_binary_symlink "$pkg" "${V}"
    return 0
}

function dnb_cppcgen() {
    local pkg="cppcgen"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    any_mode_is_set "du" "$m" && du_github "a-v-medvedev" "cppcgen" "" "$V" "$m"
    local COMMANDS=""
    local PARAMS="clean all"
    any_mode_is_set "b" "$m" && b_make "$pkg" "$V" "$COMMANDS" "$PARAMS" "$m"
    local FILES="distr/include distr/lib"
    this_mode_is_set "i" "$m" && i_direct_copy "$pkg" "$V" "$FILES" "$m"
    this_mode_is_set "i" "$m" && make_binary_symlink "$pkg" "${V}"
    return 0
}

function dnb_numactl() {
    local pkg="numactl"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    any_mode_is_set "du" "$m" && du_github "numactl" "numactl" "v" "$V" "$m"
    local COMMANDS=""
    local PARAMS="clean all"
    this_mode_is_set "b" "$m" && b_make "$pkg" "$V" "$COMMANDS" "$PARAMS" "$m"
    local FILES="libnuma.so.1 libnuma.so numa.h numaif.h"
    this_mode_is_set "i" "$m" && i_direct_copy "$pkg" "$V" "$FILES" "$m"
    this_mode_is_set "i" "$m" && make_binary_symlink "$pkg" "${V}"
    return 0
}

function dnb_scotch() {
    local pkg="scotch"
    environment_check_specific "$pkg" || fatal "${pkg}: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    du_gitlab "$pkg" "$pkg" "${pkg}-" "v$V" "$m" "gitlab.inria.fr"
    if this_mode_is_set "u" "$m"; then
        rm -rf ${pkg}-${V}.src
        mv ${pkg}-v${V}.src ${pkg}-${V}.src
        cd ${pkg}-${V}.src
	    cd src
	    [ ! -e Makefile.inc ] && ln -s Make.inc/Makefile.inc.i686_pc_linux2.shlib Makefile.inc
	    sed -i 's/LDFLAGS.*/& -lrt/;s/-DSCOTCH_PTHREAD/-DSCOTCH_DETERMINISTIC/;s/-DCOMMON_PTHREAD//' Makefile.inc
	    cd $INSTALL_DIR
    fi
    local COMMANDS=""
    COMMANDS="cd src"
    local PARAMS=""
    PARAMS="$PARAMS clean ptscotch"
    PARAMS="$PARAMS CCS=$CC CCP=$MPICC CCD=$MPICC AR=$CC"
    this_mode_is_set "b" "$m" && b_make "scotch" "${V}" "$COMMANDS" "$PARAMS" "$m"
    PARAMS="install installstub prefix=$INSTALL_DIR/${pkg}-${V}"
    this_mode_is_set "i" "$m" && b_make "scotch" "${V}" "$COMMANDS" "$PARAMS" "$m"
    this_mode_is_set "i" "$m" && make_binary_symlink "$pkg" "${V}"
    return 0
}



####

environment_check_main || fatal "Environment is not supported, exiting"
cd "$INSTALL_DIR"

PACKAGES="hypre yaml-cpp argsparser cppcgen numactl scotch"
VERSIONS="hypre:2.20.0 yaml-cpp:0.6.3 argsparser:0.0.9 cppcgen:0.0.1 numactl:1.0.2 scotch:6.1.0"
set +u
override_versions="${PACKAGE_VERSIONS}"
set -u
what_to_build=$(expand_mode_string "$*" "$PACKAGES" ":dubi")

LIST1=$(print_all_exec_files)
#---
for pkg in $PACKAGES; do
    eval dnb_${pkg} mode=$(mode_for_pkg ${pkg} "$what_to_build") version=$(version_for_pkg ${pkg} "${VERSIONS}" "${override_versions}" "${what_to_build}")
done
#---
LIST2=$(print_all_exec_files)
echo -ne "----------\nExecutables before build start: (unix-time, size, name)\n$LIST1"
echo -ne "----------\nExecutables after build finish: (unix-time, size, name)\n$LIST2"
