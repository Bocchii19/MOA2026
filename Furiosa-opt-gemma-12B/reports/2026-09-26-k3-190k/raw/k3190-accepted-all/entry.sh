#!/bin/sh
cd "$(dirname "$0")" || exit 1
cp bin_A ./k3190_A.exec || exit 126
cp bin_B ./k3190_B.exec || exit 126
chmod +x ./k3190_A.exec ./k3190_B.exec || exit 126
echo "@@AB_BEGIN 1 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./k3190_B.exec
rc=$?
echo "@@AB_END 1 B rc=$rc@@"
echo "@@AB_BEGIN 1 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./k3190_A.exec
rc=$?
echo "@@AB_END 1 A rc=$rc@@"
echo "@@AB_BEGIN 2 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./k3190_B.exec
rc=$?
echo "@@AB_END 2 B rc=$rc@@"
echo "@@AB_BEGIN 2 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./k3190_A.exec
rc=$?
echo "@@AB_END 2 A rc=$rc@@"
echo "@@AB_BEGIN 3 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./k3190_B.exec
rc=$?
echo "@@AB_END 3 B rc=$rc@@"
echo "@@AB_BEGIN 3 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./k3190_A.exec
rc=$?
echo "@@AB_END 3 A rc=$rc@@"
