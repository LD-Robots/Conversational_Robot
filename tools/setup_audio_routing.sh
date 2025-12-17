#!/usr/bin/env bash
set -euo pipefail

info() { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*"; }

require_bin() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Comanda '$1' nu există. Instaleaz-o (ex: sudo apt install pulseaudio-utils) și reia."
        exit 1
    fi
}

require_bin pactl

usage() {
    cat <<'EOT'
Folosește scriptul pentru a crea dispozitivele echo-cancel pe baza unui microfon și a unei ieșiri specificate.

Opțiuni:
  --source <nume>   Numele exact al sursei (microfon) pentru ec_mic
  --sink <nume>     Numele exact al ieșirii (difuzor) pentru ec_speaker
  -h | --help       Afișează acest mesaj și iese

Exemplu:
  ./tools/setup_audio_routing.sh \
    --source alsa_input.usb-ME6S_MS_N-B_R-UN__3db_ME6S-00.mono-fallback
EOT
}

SOURCE_OVERRIDE=""
SINK_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE_OVERRIDE="$2"
            shift 2
            ;;
        --sink)
            SINK_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Opțiune necunoscută: $1"
            usage
            exit 1
            ;;
    esac
done

DEFAULT_SINK="$(pactl info | sed -n 's/^Default Sink: //p')"
DEFAULT_SOURCE="$(pactl info | sed -n 's/^Default Source: //p')"

if [[ -z "${DEFAULT_SINK}" || -z "${DEFAULT_SOURCE}" ]]; then
    err "Nu pot citi sink/source implicit din 'pactl info'. Verifică PulseAudio/PipeWire."
    exit 1
fi

SELECTED_SINK="${SINK_OVERRIDE:-${DEFAULT_SINK}}"
SELECTED_SOURCE="${SOURCE_OVERRIDE:-${DEFAULT_SOURCE}}"

info "Default sink:   ${DEFAULT_SINK}"
info "Default source: ${DEFAULT_SOURCE}"
if [[ -n "${SINK_OVERRIDE}" ]]; then
    info "Folosește sink personalizat:   ${SELECTED_SINK}"
fi
if [[ -n "${SOURCE_OVERRIDE}" ]]; then
    info "Folosește source personalizat: ${SELECTED_SOURCE}"
fi

if pactl unload-module module-echo-cancel >/dev/null 2>&1; then
    info "Am descărcat module-echo-cancel existent."
else
    warn "N-am găsit module-echo-cancel activ (ok)."
fi

AEC_ARGS=(
    load-module module-echo-cancel
    aec_method=webrtc
    aec_args="analog_gain_control=0 digital_gain_control=0"
    use_master_format=1
    sink_master="${SELECTED_SINK}"
    source_master="${SELECTED_SOURCE}"
    sink_name=ec_speaker
    source_name=ec_mic
)

info "Încarc module-echo-cancel (webrtc)..."
pactl "${AEC_ARGS[@]}"

info "Setez ec_mic ca implicit..."
pactl set-default-source ec_mic

info "Setez ec_speaker ca implicit..."
pactl set-default-sink ec_speaker

info "Dispozitive disponibile:"
pactl list short sources | grep -Ei 'ec_mic|echo|cancel' || true
pactl list short sinks   | grep -Ei 'ec_speaker|echo|cancel' || true

cat <<'EOT'

──────────────────────────────────────────────────────────
Routing complet!

1. Rulează aplicația cu:
   PULSE_SOURCE=ec_mic PULSE_SINK=ec_speaker LOG_LEVEL=INFO LOG_DIR=logs python -m src.app

2. Dacă folosești pavucontrol, verifică că procesul python apare cu:
   - Playback -> ec_speaker
   - Recording -> ec_mic

3. Dacă vrei să revii la dispozitivele implicite originale:
   pactl unload-module module-echo-cancel
   pactl set-default-source "${DEFAULT_SOURCE}"
   pactl set-default-sink "${DEFAULT_SINK}"
──────────────────────────────────────────────────────────
EOT