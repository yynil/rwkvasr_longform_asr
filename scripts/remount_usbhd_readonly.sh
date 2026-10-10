#!/usr/bin/env bash
set -euo pipefail

MOUNT_POINT="${1:-/media/usbhd}"
FILESYSTEM_UUID="${USBHD_UUID:-ed2bac74-5c33-4d70-915e-bea717179fca}"
VERIFY_FILE="${USBHD_VERIFY_FILE:-common_voice_22/webdataset/zh-CN/shard_000000.tar}"
DEVICE_LINK="/dev/disk/by-uuid/${FILESYSTEM_UUID}"

log() {
  printf '[usbhd-remount] %s\n' "$*"
}

if [[ ! -e "${DEVICE_LINK}" ]]; then
  log "device is unavailable: ${DEVICE_LINK}"
  lsblk -o NAME,MODEL,SERIAL,SIZE,FSTYPE,MOUNTPOINTS,RO,STATE,TRAN
  exit 1
fi

EXPECTED_DEVICE="$(readlink -f "${DEVICE_LINK}")"
EXPECTED_MAJ_MIN="$(lsblk -dn -o MAJ:MIN "${EXPECTED_DEVICE}" | tr -d ' ')"
if [[ -z "${EXPECTED_MAJ_MIN}" ]]; then
  log "could not resolve MAJ:MIN for ${EXPECTED_DEVICE}"
  exit 1
fi

verify_mount() {
  local current_maj_min current_source current_fstype current_options verify_path
  if ! mountpoint -q "${MOUNT_POINT}"; then
    return 1
  fi
  current_maj_min="$(findmnt -n -o MAJ:MIN --target "${MOUNT_POINT}" | tr -d ' ')"
  current_source="$(findmnt -n -o SOURCE --target "${MOUNT_POINT}")"
  current_fstype="$(findmnt -n -o FSTYPE --target "${MOUNT_POINT}")"
  current_options="$(findmnt -n -o OPTIONS --target "${MOUNT_POINT}")"
  log "mounted source=${current_source} maj:min=${current_maj_min} fstype=${current_fstype} options=${current_options}"
  if [[ "${current_maj_min}" != "${EXPECTED_MAJ_MIN}" || "${current_fstype}" != "xfs" ]]; then
    return 1
  fi
  if [[ ",${current_options}," != *,ro,* || ",${current_options}," != *,norecovery,* ]]; then
    log "mount is not read-only XFS norecovery"
    return 1
  fi
  verify_path="${MOUNT_POINT}/${VERIFY_FILE}"
  if [[ ! -f "${verify_path}" ]]; then
    log "verification file is unavailable: ${verify_path}"
    return 1
  fi
  log "reading first 8 MiB from ${verify_path}"
  if ! dd if="${verify_path}" bs=1M count=8 iflag=fullblock status=none | sha256sum; then
    log "verification read failed"
    return 1
  fi
}

log "expected device=${EXPECTED_DEVICE} maj:min=${EXPECTED_MAJ_MIN} uuid=${FILESYSTEM_UUID}"
if verify_mount; then
  log "mount and real-file read are healthy; no remount needed"
  exit 0
fi

if mountpoint -q "${MOUNT_POINT}"; then
  log "unmounting stale or unhealthy mount ${MOUNT_POINT}"
  if ! sudo umount "${MOUNT_POINT}"; then
    log "normal unmount failed; using lazy unmount for the stale mount"
    sudo umount -l "${MOUNT_POINT}"
  fi
fi

sudo mkdir -p "${MOUNT_POINT}"
log "mounting ${DEVICE_LINK} at ${MOUNT_POINT} as ro,norecovery"
sudo mount -t xfs -o ro,norecovery "${DEVICE_LINK}" "${MOUNT_POINT}"

if ! verify_mount; then
  log "post-mount validation failed"
  findmnt "${MOUNT_POINT}" -o MAJ:MIN,SOURCE,TARGET,FSTYPE,OPTIONS || true
  exit 1
fi

log "remount and real-file read completed successfully"
