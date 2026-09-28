#!/bin/sh
cd "$(dirname "$0")" || exit 1
chmod +x ./bin_A ./bin_B 2>/dev/null
for r in $(seq 1 3); do
  for arm in A B; do
    echo "@@AB_BEGIN $r $arm@@"
    GEMMA4_ONLY="decoder_feedforward" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./bin_$arm
    echo "@@AB_END $r $arm rc=$?@@"
  done
done
