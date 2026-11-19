#!/bin/sh
# Installs the standalone `eventcv` command from the GitHub release binaries:
#
#   curl -LsSf https://github.com/EventLAB-Team/eventcv/releases/latest/download/install.sh | sh
#
# EVENTCV_VERSION=X.Y.Z pins a release (default: the latest), EVENTCV_INSTALL_DIR picks where the
# binary goes (default: ~/.local/bin), and EVENTCV_DOWNLOAD_URL replaces the release URL entirely
# (a mirror, or `file://…` — which is how CI tests this script against the archives it just built).
# Upgrading is running it again; uninstalling is deleting the one file it prints.
set -eu

repo="https://github.com/EventLAB-Team/eventcv"

fail() {
    echo "eventcv install: $*" >&2
    exit 1
}

case "$(uname -s)" in
    Linux) os="unknown-linux-gnu" ;;
    Darwin) os="apple-darwin" ;;
    *) fail "no prebuilt binary for $(uname -s); on Windows use install.ps1, elsewhere build from source (see docs.eventcv.net)" ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) arch="x86_64" ;;
    aarch64 | arm64) arch="aarch64" ;;
    *) fail "no prebuilt binary for $(uname -m) on $(uname -s)" ;;
esac
# A shell running under Rosetta reports x86_64; the native build is the one to have.
if [ "$os" = "apple-darwin" ] && [ "$arch" = "x86_64" ] &&
    [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
    arch="aarch64"
fi
archive="eventcv-$arch-$os.tar.gz"

if [ -n "${EVENTCV_DOWNLOAD_URL:-}" ]; then
    base="$EVENTCV_DOWNLOAD_URL"
elif [ -n "${EVENTCV_VERSION:-}" ]; then
    base="$repo/releases/download/v${EVENTCV_VERSION#v}"
else
    base="$repo/releases/latest/download"
fi
install_dir="${EVENTCV_INSTALL_DIR:-$HOME/.local/bin}"

download() {
    if command -v curl >/dev/null 2>&1; then
        curl --proto '=https,file' --tlsv1.2 -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$1" -O "$2"
    else
        fail "needs curl or wget"
    fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

echo "downloading $base/$archive"
download "$base/$archive" "$tmp/$archive" || fail "could not download $base/$archive"
download "$base/$archive.sha256" "$tmp/$archive.sha256" || fail "could not download the checksum"

expected="$(cut -d ' ' -f 1 <"$tmp/$archive.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$tmp/$archive" | cut -d ' ' -f 1)"
else
    actual="$(shasum -a 256 "$tmp/$archive" | cut -d ' ' -f 1)"
fi
[ "$expected" = "$actual" ] || fail "checksum mismatch for $archive (expected $expected, got $actual)"

tar -xzf "$tmp/$archive" -C "$tmp"
mkdir -p "$install_dir"
# Through a temporary name, so an `eventcv` that is running right now is replaced, not truncated.
cp "$tmp/eventcv" "$install_dir/.eventcv.new"
chmod 755 "$install_dir/.eventcv.new"
mv -f "$install_dir/.eventcv.new" "$install_dir/eventcv"

"$install_dir/eventcv" --version | head -n 1 | sed "s|^|installed |; s|$| to $install_dir/eventcv|"

case ":$PATH:" in
    *":$install_dir:"*)
        found="$(command -v eventcv || true)"
        # A pip-installed `eventcv` earlier on PATH would quietly answer instead of this one.
        if [ -n "$found" ] && [ "$found" != "$install_dir/eventcv" ]; then
            echo "note: $found comes first on PATH, so \`eventcv\` runs that one, not this"
        fi
        ;;
    *) echo "note: $install_dir is not on PATH; add it to your shell profile, e.g. export PATH=\"$install_dir:\$PATH\"" ;;
esac
if [ "$os" = "unknown-linux-gnu" ]; then
    echo "note: USB cameras need udev rules for non-root access on Linux (see Streaming at https://docs.eventcv.net)"
fi
