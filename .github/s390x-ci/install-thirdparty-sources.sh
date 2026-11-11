#!/usr/bin/env bash

set -euxo pipefail

: "${JSON_VERSION:?JSON_VERSION must be set}"
: "${SLEEF_COMMIT:?SLEEF_COMMIT must be set}"

# TRITON_OFFLINE_BUILD stops the build from downloading these.
tmp="$(mktemp -d)"
curl --fail --location --silent --show-error --output "${tmp}/include.zip" \
    "https://github.com/nlohmann/json/releases/download/${JSON_VERSION}/include.zip"
mkdir -p /opt/json
unzip -q "${tmp}/include.zip" -d /opt/json
test -f /opt/json/include/nlohmann/json.hpp
rm -rf "${tmp}"

git clone --depth 1 --revision="${SLEEF_COMMIT}" https://github.com/shibatch/sleef.git /opt/sleef
rm -rf /opt/sleef/.git
