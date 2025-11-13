#!/bin/bash

mkdir kokkos-install
export KOKKOS_VERSION=4.7.01  # Replace with the actual version
export KOKKOS_DOWNLOAD_URL=https://github.com/kokkos/kokkos/releases/download/${KOKKOS_VERSION}
curl -sLO ${KOKKOS_DOWNLOAD_URL}/kokkos-${KOKKOS_VERSION}.tar.gz
curl -sLO ${KOKKOS_DOWNLOAD_URL}/kokkos-${KOKKOS_VERSION}-SHA-256.txt

RESULT="$(grep kokkos-${KOKKOS_VERSION}.tar.gz kokkos-${KOKKOS_VERSION}-SHA-256.txt | shasum -c)"

if [ "$RESULT" = "kokkos-${KOKKOS_VERSION}.tar.gz: OK" ]; then
    tar -xzvf kokkos-${KOKKOS_VERSION}.tar.gz
    cd kokkos-${KOKKOS_VERSION}
    cmake -B builddir \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_INSTALL_PREFIX=./../kokkos-install/ \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF
    cmake --build builddir
    cmake --install builddir
else
    echo "Failure"
fi