#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/stages/stage179_usbhd_dedup_online_ctc_alignment}"
DEST_ROOT="${DEST_ROOT:-${HOME}/rwkvasr_data/stage211_full_curriculum}"
RSYNC_BWLIMIT_KIB="${RSYNC_BWLIMIT_KIB:-20000}"

STAGES=(
  stage179b_medium_dedup_audio_only_online_ctc
  stage179c_hard_dedup_audio_only_online_ctc
  stage179d_long_dedup_audio_only_online_ctc
)

for stage in "${STAGES[@]}"; do
  source="${SOURCE_ROOT}/${stage}/webdataset_buckets_audio_text/"
  destination="${DEST_ROOT}/${stage}/webdataset_buckets_audio_text/"
  if [[ ! -s "${source}/manifest.json" ]]; then
    echo "source manifest unavailable: ${source}/manifest.json" >&2
    exit 1
  fi
  mkdir -p "${destination}"
  printf '[stage211-metadata] copy stage=%s source=%s destination=%s\n' \
    "${stage}" "${source}" "${destination}"
  rsync \
    --archive \
    --partial \
    --human-readable \
    --info=progress2 \
    --bwlimit="${RSYNC_BWLIMIT_KIB}" \
    "${source}" \
    "${destination}"
  source_sha="$(sha256sum "${source}/manifest.json" | awk '{print $1}')"
  destination_sha="$(sha256sum "${destination}/manifest.json" | awk '{print $1}')"
  if [[ "${source_sha}" != "${destination_sha}" ]]; then
    echo "manifest SHA-256 mismatch after copy: ${stage}" >&2
    exit 1
  fi
  printf '[stage211-metadata] verified stage=%s manifest_sha256=%s\n' \
    "${stage}" "${destination_sha}"
done

printf '[stage211-metadata] complete destination=%s\n' "${DEST_ROOT}"
