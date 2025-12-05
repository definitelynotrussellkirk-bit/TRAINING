#!/bin/bash
# setup_synology.sh - Configure Synology NAS mounts for the Realm
#
# Usage:
#   ./scripts/setup_synology.sh           # Interactive setup
#   ./scripts/setup_synology.sh --check   # Check mount status only
#   ./scripts/setup_synology.sh --mount   # Mount (if in fstab)
#   ./scripts/setup_synology.sh --unmount # Unmount

set -e

# Configuration from hosts.json
NAS_IP="192.168.30.15"
NAS_USER="russ"

# Local mount points
MOUNT_BASE="/mnt/synology"
MOUNT_DATA="${MOUNT_BASE}/data"
MOUNT_BACKUP="${MOUNT_BASE}/backup"
MOUNT_ARCHIVE="${MOUNT_BASE}/archive"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[-]${NC} $1"
}

check_mount() {
    local mount_point="$1"
    if mountpoint -q "$mount_point" 2>/dev/null; then
        return 0
    fi
    return 1
}

check_all_mounts() {
    echo ""
    echo "=== Synology Mount Status ==="
    echo ""

    local all_ok=true

    for mount in "$MOUNT_DATA" "$MOUNT_BACKUP" "$MOUNT_ARCHIVE"; do
        if check_mount "$mount"; then
            local usage=$(df -h "$mount" 2>/dev/null | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
            print_success "$mount: MOUNTED - $usage"
        else
            if [ -d "$mount" ]; then
                print_warning "$mount: NOT MOUNTED (directory exists)"
            else
                print_error "$mount: NOT MOUNTED (directory missing)"
            fi
            all_ok=false
        fi
    done

    echo ""

    if $all_ok; then
        print_success "All Synology mounts are active"
        return 0
    else
        print_warning "Some mounts are not active"
        return 1
    fi
}

test_connectivity() {
    print_status "Testing connectivity to NAS at $NAS_IP..."

    if ping -c 1 -W 2 "$NAS_IP" > /dev/null 2>&1; then
        print_success "NAS is reachable"
        return 0
    else
        print_error "Cannot reach NAS at $NAS_IP"
        return 1
    fi
}

check_nfs_support() {
    print_status "Checking NFS client support..."

    if command -v mount.nfs &> /dev/null; then
        print_success "NFS client is installed"
        return 0
    else
        print_warning "NFS client not found. Install with: sudo apt install nfs-common"
        return 1
    fi
}

check_cifs_support() {
    print_status "Checking CIFS/SMB client support..."

    if command -v mount.cifs &> /dev/null; then
        print_success "CIFS client is installed"
        return 0
    else
        print_warning "CIFS client not found. Install with: sudo apt install cifs-utils"
        return 1
    fi
}

create_mount_points() {
    print_status "Creating mount point directories..."

    sudo mkdir -p "$MOUNT_DATA" "$MOUNT_BACKUP" "$MOUNT_ARCHIVE"
    sudo chown "$USER:$USER" "$MOUNT_BASE"

    print_success "Mount points created at $MOUNT_BASE"
}

setup_credentials() {
    local creds_file="/etc/synology-credentials"

    if [ -f "$creds_file" ]; then
        print_success "Credentials file already exists at $creds_file"
        return 0
    fi

    echo ""
    print_status "Setting up credentials for Synology access..."
    echo "Enter the password for NAS user '$NAS_USER':"
    read -s nas_password
    echo ""

    sudo bash -c "cat > $creds_file << EOF
username=$NAS_USER
password=$nas_password
EOF"
    sudo chmod 600 "$creds_file"

    print_success "Credentials saved to $creds_file"
}

generate_fstab_entries() {
    echo ""
    echo "=== fstab entries for Synology mounts ==="
    echo ""
    echo "# Add these lines to /etc/fstab for persistent mounts:"
    echo ""

    # NFS version (preferred for performance)
    echo "# Option 1: NFS (recommended for Linux)"
    echo "//${NAS_IP}/data/llm_training      ${MOUNT_DATA}    nfs  defaults,_netdev,nofail  0  0"
    echo "//${NAS_IP}/backup/llm_training    ${MOUNT_BACKUP}  nfs  defaults,_netdev,nofail  0  0"
    echo "//${NAS_IP}/archive/llm_training   ${MOUNT_ARCHIVE} nfs  defaults,_netdev,nofail  0  0"
    echo ""

    # CIFS version (alternative)
    echo "# Option 2: CIFS/SMB (works with Windows shares)"
    echo "//${NAS_IP}/data/llm_training      ${MOUNT_DATA}    cifs  credentials=/etc/synology-credentials,uid=$(id -u),gid=$(id -g),_netdev,nofail  0  0"
    echo "//${NAS_IP}/backup/llm_training    ${MOUNT_BACKUP}  cifs  credentials=/etc/synology-credentials,uid=$(id -u),gid=$(id -g),_netdev,nofail  0  0"
    echo "//${NAS_IP}/archive/llm_training   ${MOUNT_ARCHIVE} cifs  credentials=/etc/synology-credentials,uid=$(id -u),gid=$(id -g),_netdev,nofail  0  0"
    echo ""
}

mount_all() {
    print_status "Mounting all Synology shares..."

    for mount in "$MOUNT_DATA" "$MOUNT_BACKUP" "$MOUNT_ARCHIVE"; do
        if check_mount "$mount"; then
            print_success "$mount already mounted"
        else
            print_status "Mounting $mount..."
            if sudo mount "$mount" 2>/dev/null; then
                print_success "Mounted $mount"
            else
                print_error "Failed to mount $mount (not in fstab or share unavailable)"
            fi
        fi
    done
}

unmount_all() {
    print_status "Unmounting all Synology shares..."

    for mount in "$MOUNT_DATA" "$MOUNT_BACKUP" "$MOUNT_ARCHIVE"; do
        if check_mount "$mount"; then
            print_status "Unmounting $mount..."
            sudo umount "$mount"
            print_success "Unmounted $mount"
        else
            print_warning "$mount not mounted"
        fi
    done
}

interactive_setup() {
    echo ""
    echo "=========================================="
    echo "  Synology NAS Setup for Realm of Training"
    echo "=========================================="
    echo ""

    # Step 1: Test connectivity
    if ! test_connectivity; then
        print_error "Cannot proceed without NAS connectivity"
        exit 1
    fi

    # Step 2: Check client support
    echo ""
    local use_nfs=false
    local use_cifs=false

    if check_nfs_support; then
        use_nfs=true
    fi

    if check_cifs_support; then
        use_cifs=true
    fi

    if ! $use_nfs && ! $use_cifs; then
        echo ""
        print_error "No mount client available. Install one of:"
        echo "  sudo apt install nfs-common    # For NFS"
        echo "  sudo apt install cifs-utils    # For CIFS/SMB"
        exit 1
    fi

    # Step 3: Create mount points
    echo ""
    create_mount_points

    # Step 4: Setup credentials (for CIFS)
    if $use_cifs; then
        setup_credentials
    fi

    # Step 5: Show fstab entries
    generate_fstab_entries

    echo ""
    print_status "Next steps:"
    echo "  1. Enable NFS or SMB sharing on your Synology for these paths:"
    echo "     - /volume1/data/llm_training"
    echo "     - /volume1/backup/llm_training"
    echo "     - /volume1/archive/llm_training"
    echo ""
    echo "  2. Add the appropriate fstab entries shown above"
    echo "     sudo nano /etc/fstab"
    echo ""
    echo "  3. Mount the shares:"
    echo "     sudo mount -a"
    echo ""
    echo "  4. Verify mounts:"
    echo "     ./scripts/setup_synology.sh --check"
    echo ""
}

update_storage_config() {
    print_status "Updating storage_zones.json with local mount paths..."

    local config_file="/home/russ/Desktop/TRAINING/config/storage_zones.json"

    # This would need to update the config - for now just show what needs to change
    echo ""
    echo "=== Update storage_zones.json ==="
    echo ""
    echo "Change the warm/cold zone roots from:"
    echo '  "synology_data": "/volume1/data/llm_training"'
    echo ""
    echo "To:"
    echo "  \"synology_data\": \"${MOUNT_DATA}\""
    echo "  \"synology_backup\": \"${MOUNT_BACKUP}\""
    echo "  \"synology_archive\": \"${MOUNT_ARCHIVE}\""
    echo ""
}

# Main
case "${1:-}" in
    --check)
        check_all_mounts
        ;;
    --mount)
        mount_all
        check_all_mounts
        ;;
    --unmount)
        unmount_all
        ;;
    --fstab)
        generate_fstab_entries
        ;;
    --config)
        update_storage_config
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  (none)      Interactive setup wizard"
        echo "  --check     Check current mount status"
        echo "  --mount     Mount shares (must be in fstab)"
        echo "  --unmount   Unmount all Synology shares"
        echo "  --fstab     Show fstab entries to add"
        echo "  --config    Show config updates needed"
        echo "  --help      Show this help"
        ;;
    *)
        interactive_setup
        ;;
esac
