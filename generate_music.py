"""
Music generation using ACE-Step 1.5.

Setup:
    git clone https://github.com/ace-step/ACE-Step-1.5.git
    cd ACE-Step-1.5
    uv sync

Run (from this script's directory or any directory):
    cd ~/dev/ACE-Step-1.5
    uv run python /path/to/generate_music.py

Models are downloaded automatically to ACE-Step-1.5/checkpoints/ on first run.

VRAM requirements:
    acestep-v15-turbo (current): ~6–8 GB — works on most modern GPUs.
    acestep-v15-xl-sft (4B):     ~20–24 GB — produces noise, not recommended.
"""

import math
import os
import sys

import soundfile as sf
import torch
import torchaudio

# Path to the cloned ACE-Step-1.5 repo — adjust if cloned elsewhere.
ACESTEP_DIR = os.path.expanduser("~/dev/ACE-Step-1.5")
sys.path.insert(0, ACESTEP_DIR)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_TRACK = os.path.join(_SCRIPT_DIR, "theme.wav")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "generated_tracks")

# Default crossfade. Overridden per-track via "crossfade_sec" in MUSIC_TRACKS.
# Longer = smoother loop; shorter = tighter rhythmic loop.
DEFAULT_CROSSFADE_SEC = 2.0

# Turbo model: 8 steps, no CFG, no shift needed. Fast and reliable.
# sft/xl-sft models consistently produce over-amplified noise due to a CFG
# implementation issue in their base model class; turbo is the confirmed-working path.
INFERENCE_STEPS = 8    # turbo model hard-caps at 8; higher values are silently clamped
GUIDANCE_SCALE = 0.0   # ignored by turbo; 0 avoids any accidental CFG path
SHIFT = 1.0            # ignored by turbo; default value

# How much the source melody shapes each track.
# 0.4 keeps enough harmonic character to sound like the same game
# without over-constraining the model's timbre choices.
AUDIO_COVER_STRENGTH = 0.4

# Detected from theme.wav via Krumhansl-Schmuckler chroma analysis.
# Locking these across all tracks is the primary cohesion mechanism.
THEME_KEY = "A Major"
THEME_BPM = 100        # theme measured at 99.4; round to 100
THEME_TIME_SIG = "4"   # 4/4

# The model normalizes output to -1.0 dBFS peak, which is loud for a game loop.
# -9 dB gives comfortable headroom for mixing and looping.
OUTPUT_LEVEL_DB = -9.0

# Turbo model: 2B params, ~6–8 GB VRAM, confirmed musical output.
DIT_MODEL = "acestep-v15-turbo"

# LM planner is silently bypassed for cover tasks regardless.
LM_MODEL_SIZE = None

# ---------------------------------------------------------------------------
# Track definitions — one per game phase
# ---------------------------------------------------------------------------
#
# caption   : Rich natural-language description used by both the LM planner and
#             the DiT. More specificity (instruments, texture, tempo feel, mood)
#             = more accurate results.
# crossfade_sec : (optional) override the default loop crossfade window.
#             Ice/ambient tracks get longer windows; rhythmic tracks get shorter.

MUSIC_TRACKS = {

    # Phase 1A — Levels 1-10: First steps, tiny grids.
    "1A - First Steps": {
        "caption": "video game music, acoustic piano, marimba, harp, slow tempo, gentle, whimsical, hopeful, 8-bit inspired",
        "bpm": 80,
        "crossfade_sec": 2.5,
    },

    # Phase 1B — Levels 11-35: Ramps introduced, grids growing.
    "1B - Rising Paths": {
        "caption": "video game music, piano, flute, legato strings, moderate upbeat tempo, flowing, playful, curious, optimistic, platformer-style",
        "bpm": 100,
        "crossfade_sec": 2.0,
    },

    # Phase 1C — Levels 36-75: Ramp mastery, maze archetypes.
    "1C - The Winding Way": {
        "caption": "video game music, acoustic guitar, flute, light strings, harp, moderate tempo, flowing, curious, mysterious, adventure exploration",
        "bpm": 100,
        "crossfade_sec": 2.0,
    },

    # Phase 1D — Levels 76-100: Ramp exam, open fields.
    "1D - Open Country": {
        "caption": "acoustic folk soundtrack, acoustic piano, soft pizzicato strings, gentle flute, relaxed tempo, airy, pastoral, carefree, soft dynamics, quiet, countryside walk",
        "bpm": 90,
        "crossfade_sec": 3.0,
    },

    # Phase 2A — Levels 101-130: Switches introduced.
    "2A - Circuits Awaken": {
        "caption": "video game music, electric piano, synth bells, electronic percussion, bass, moderate tempo, alert, mechanical, 3D platformer-style",
        "bpm": 100,
        "crossfade_sec": 1.5,
    },

    # Phase 2B — Levels 131-190: Switch sequencing, 2-5 switches.
    "2B - Chain Logic": {
        "caption": "video game music, synth arpeggios, electronic drums, lead synthesizer, bass pulse, upbeat driven tempo, energized, determined, adventure-style",
        "bpm": 110,
        "crossfade_sec": 1.5,
    },

    # Phase 2C — Levels 191-250: Switch mastery, large grids.
    "2C - The Grid": {
        "caption": "video game music, synth bass, arpeggiated synthesizer, electronic drums, driving tempo, focused, energized, mechanical, sci-fi electronic",
        "bpm": 120,
        "crossfade_sec": 1.5,
    },

    # Phase 3A — Levels 251-270: Ice introduced.
    "3A - First Frost": {
        "caption": "video game music, celeste, music box, acoustic piano, gentle harp, very slow tempo, soft, icy, peaceful, delicate, wintry ambient",
        "bpm": 70,
        "crossfade_sec": 3.5,
    },

    # Phase 3B — Levels 271-325: Ice block puzzles.
    "3B - Cold Calculation": {
        "caption": "video game music, vibraphone, celeste, light strings, piano, slow deliberate tempo, cool, thoughtful, precise, elegant, icy puzzle",
        "bpm": 80,
        "crossfade_sec": 3.0,
    },

    # Phase 3C — Levels 326-360: Ice mastery.
    "3C - Frozen Labyrinth": {
        "caption": "video game music, celeste, bells, sustained strings, piano, moderate tempo, tense, cold, determined, icy dungeon",
        "bpm": 100,
        "crossfade_sec": 2.5,
    },

    # Phase 4 — Levels 361-500: Endgame.
    "4 - The Grand Design": {
        "caption": "acoustic piano, steady beat, harp, marimba, acoustic guitar, moderate flowing tempo, warm, bright, hopeful, peaceful, joyful, grand finale",
        "bpm": 100,
        "crossfade_sec": 3.0,
    },
}

# Phases to regenerate — only tracks that didn't land on attempt 1.
# Set to None to regenerate all phases.
REGEN_PHASES = {
    "1D - Open Country",
    "4 - The Grand Design",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def get_reference_duration(path: str) -> float:
    return sf.info(path).duration


def make_seamless_loop(audio: torch.Tensor, sr: int, crossfade_sec: float) -> torch.Tensor:
    """
    Overlay the track's own head onto its tail with a linear crossfade.

    At the loop point the engine jumps from the last sample back to sample 0.
    With this blend the tail is already fading into the head's content, so the
    jump is inaudible. Works best when the generated phrase is musically coherent
    at both ends — ACE-Step's structure-aware LM planner helps ensure this.
    """
    n = min(int(crossfade_sec * sr), audio.shape[-1] // 3)
    if n == 0:
        return audio

    fade_out = torch.linspace(1.0, 0.0, n, device=audio.device)
    fade_in  = torch.linspace(0.0, 1.0, n, device=audio.device)

    result = audio.clone()
    result[..., -n:] = audio[..., -n:] * fade_out + audio[..., :n] * fade_in
    return result


def load_handlers(device: str):
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=ACESTEP_DIR,
        config_path=DIT_MODEL,
        device=device,
        prefer_source="modelscope",  # HuggingFace stalls on large shards
    )

    llm = None
    if LM_MODEL_SIZE is not None:
        llm = LLMHandler()
        llm.initialize(
            checkpoint_dir=os.path.join(ACESTEP_DIR, "checkpoints"),
            lm_model_path=LM_MODEL_SIZE,
            backend="pytorch",
            device=device,
        )

    return dit, llm


def generate_track(
    dit: AceStepHandler,
    llm,
    name: str,
    cfg: dict,
    ref_duration: float,
) -> None:
    crossfade_sec = cfg.get("crossfade_sec", DEFAULT_CROSSFADE_SEC)
    gen_duration = math.ceil(ref_duration)

    params = GenerationParams(
        task_type="cover",
        src_audio=REFERENCE_TRACK,
        audio_cover_strength=AUDIO_COVER_STRENGTH,
        caption=cfg["caption"],
        instrumental=True,
        duration=gen_duration,
        inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        shift=SHIFT,
        thinking=llm is not None,
        seed=-1,
        bpm=cfg.get("bpm", THEME_BPM),
        keyscale=THEME_KEY,
        timesignature=THEME_TIME_SIG,
    )

    result = generate_music(
        dit, llm, params,
        GenerationConfig(batch_size=1, audio_format="wav"),
        save_dir=None,
    )

    if not result.success:
        print(f"  FAILED: {result.error}")
        return

    audio: torch.Tensor = result.audios[0]["tensor"]       # [2, samples], float32
    sr: int             = result.audios[0]["sample_rate"]  # 48 000 Hz stereo

    target_samples = int(ref_duration * sr)
    audio = audio[..., :target_samples]
    audio = make_seamless_loop(audio, sr, crossfade_sec)
    audio = audio * (10 ** (OUTPUT_LEVEL_DB / 20))

    out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    torchaudio.save(
        out_path,
        audio.cpu(),
        sample_rate=sr,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=16,
    )
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device    : {device}")

    ref_duration = get_reference_duration(REFERENCE_TRACK)
    print(f"Reference : {REFERENCE_TRACK}  ({ref_duration:.2f}s)")
    print(f"Model     : {DIT_MODEL}  +  LM {LM_MODEL_SIZE}\n")

    dit, llm = load_handlers(device)

    tracks = (
        {k: v for k, v in MUSIC_TRACKS.items() if k in REGEN_PHASES}
        if REGEN_PHASES is not None
        else MUSIC_TRACKS
    )
    for name, cfg in tracks.items():
        print(f"Generating: {name}")
        generate_track(dit, llm, name, cfg, ref_duration)

    print("\nAll tracks done.")


if __name__ == "__main__":
    main()
