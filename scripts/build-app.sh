#!/usr/bin/env bash
# Builds "ML Workbench.app" from this checkout, and with --install copies it
# to /Applications (or $INSTALL_DIR) and prepares its Python environment.
#
#   scripts/build-app.sh            # build/ML Workbench.app
#   scripts/build-app.sh --install  # ...and install it
#
# The app carries python/ and content/ in Contents/Resources. Because the
# bundle is signed and shouldn't change after that, the sidecar's virtualenv
# and bytecode live in ~/Library/Application Support/ML Workbench instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="ML Workbench"
EXECUTABLE="MLWorkbench"
OUT="$ROOT/build"
APP="$OUT/$APP_NAME.app"
INSTALL_DIR="${INSTALL_DIR:-/Applications}"
STATE_DIR="$HOME/Library/Application Support/$APP_NAME"

install=0
for arg in "$@"; do
  case "$arg" in
    --install) install=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }

step "Building $EXECUTABLE (release)"
cd "$ROOT"
swift build -c release --product "$EXECUTABLE"
BIN="$(swift build -c release --show-bin-path)/$EXECUTABLE"

step "Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/$EXECUTABLE"

version="$(sed -n 's/^version = "\(.*\)"/\1/p' python/pyproject.toml | head -1)"
build="$(git rev-list --count HEAD 2>/dev/null || echo 1)"
sed -e "s/__VERSION__/${version:-0.1.0}/" -e "s/__BUILD__/$build/" \
  packaging/Info.plist > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

# Python sidecar and lessons, minus anything local to this checkout.
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'tests' \
  python/ "$APP/Contents/Resources/python/"
rsync -a --delete --exclude '__pycache__' --exclude '.pytest_cache' \
  content/ "$APP/Contents/Resources/content/"

step "Making the icon"
iconset="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
  sips -z $size $size packaging/AppIcon.png --out "$iconset/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z $double $double packaging/AppIcon.png --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$iconset")"

step "Signing (ad hoc)"
codesign --force --deep --sign - --identifier com.coltonweaver.MLWorkbench "$APP"
codesign --verify --strict "$APP"

echo "Built $APP"
[ "$install" = 1 ] || exit 0

step "Installing to $INSTALL_DIR"
DEST="$INSTALL_DIR/$APP_NAME.app"
if pgrep -xq "$EXECUTABLE"; then
  echo "Quitting the running $APP_NAME"
  osascript -e "quit app \"$APP_NAME\"" >/dev/null 2>&1 || true
  sleep 1
fi
mkdir -p "$INSTALL_DIR"
rm -rf "$DEST"
ditto "$APP" "$DEST"

# Build the Python environment now so the first launch doesn't wait on a
# torch download. The app finds the same environment through
# UV_PROJECT_ENVIRONMENT (see SidecarLaunch.stateDirectory).
UV="${WORKBENCH_UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [ -x "$candidate" ] && UV="$candidate" && break
  done
fi
if [ -n "$UV" ]; then
  step "Preparing the Python environment in $STATE_DIR"
  mkdir -p "$STATE_DIR"
  UV_PROJECT_ENVIRONMENT="$STATE_DIR/python-env" \
    "$UV" sync --frozen --project "$DEST/Contents/Resources/python"
  # Nothing above may write into the bundle; a broken seal would show here.
  codesign --verify --strict "$DEST"
else
  echo "warning: uv not found; install it (brew install uv) before opening the app" >&2
fi

echo
echo "Installed $DEST"
echo "Open it with:  open \"$DEST\""
