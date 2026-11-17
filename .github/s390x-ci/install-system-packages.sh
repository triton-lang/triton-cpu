#!/usr/bin/env bash

set -euxo pipefail

# Build toolchain plus the deps of the s390x source builds (LLVM, scipy,
# pillow, matplotlib). Ubuntu has no libomp-dev for s390x.
packages=(
    build-essential
    ca-certificates
    ccache
    clang
    cmake
    curl
    file
    gfortran
    git
    jq
    libedit-dev
    libffi-dev
    libfreetype-dev
    libjpeg-turbo8-dev
    liblcms2-dev
    libopenblas-dev
    libpng-dev
    libtiff-dev
    libwebp-dev
    libxml2-dev
    lld
    ninja-build
    pkg-config
    python3-dev
    python3-pip
    python3-venv
    unzip
    xz-utils
    zip
    zlib1g-dev
)

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"
rm -rf /var/lib/apt/lists/*
