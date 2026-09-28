#!/bin/sh
cd "$(dirname "$0")" || exit 1
cp bin_A ./k3190_A.exec || exit 126
cp bin_B ./k3190_B.exec || exit 126
chmod +x ./k3190_A.exec ./k3190_B.exec || exit 126
echo "@@AB_BEGIN 1 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=trace GEMMA4_DUMP_SPANS=1 ./k3190_A.exec
rc=$?
echo "@@AB_END 1 A rc=$rc@@"
echo "@@AB_BEGIN 1 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=trace GEMMA4_DUMP_SPANS=1 ./k3190_B.exec
rc=$?
echo "@@AB_END 1 B rc=$rc@@"
