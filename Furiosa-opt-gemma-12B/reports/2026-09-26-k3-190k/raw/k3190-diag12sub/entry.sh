#!/bin/sh
cd "$(dirname "$0")" || exit 1
cp bin_A /tmp/k3190_A
cp bin_B /tmp/k3190_B
chmod +x /tmp/k3190_A /tmp/k3190_B
echo "@@AB_BEGIN 1 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info /tmp/k3190_A
rc=$?
echo "@@AB_END 1 A rc=$rc@@"
echo "@@AB_BEGIN 1 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info /tmp/k3190_B
rc=$?
echo "@@AB_END 1 B rc=$rc@@"
echo "@@AB_BEGIN 2 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info /tmp/k3190_A
rc=$?
echo "@@AB_END 2 A rc=$rc@@"
echo "@@AB_BEGIN 2 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info /tmp/k3190_B
rc=$?
echo "@@AB_END 2 B rc=$rc@@"
echo "@@AB_BEGIN 3 A@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info /tmp/k3190_A
rc=$?
echo "@@AB_END 3 A rc=$rc@@"
echo "@@AB_BEGIN 3 B@@"
GEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info /tmp/k3190_B
rc=$?
echo "@@AB_END 3 B rc=$rc@@"
