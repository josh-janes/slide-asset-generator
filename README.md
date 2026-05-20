<p align="center">
  <img src="final/banner.png" alt="AI Game Music Pipeline" />
</p>

# AI Game Music Pipeline

Stable diffusion pipeline for generating cohesive, seamless-looping game soundtracks from a single reference melody. Built on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), a 2B-parameter diffusion transformer for audio generation.

**Made for [Slide 2.0](https://play.google.com/store/apps/details?id=com.slidesequel)** — a relaxing puzzle game on [Google Play](https://play.google.com/store/apps/details?id=com.slidesequel).

## What It Does

Given a reference track (`theme.wav`), the pipeline generates 10 distinct game music tracks spanning 4 difficulty phases. Each track inherits harmonic character from the source melody while adapting instrumentation, tempo, and mood to match its game phase.

### Demo

**1A — First Steps** (acoustic piano, marimba, harp, gentle and whimsical)

[▶ Play demo](final/1A%20-%20First%20Steps.mp3)

## Architecture

```
theme.wav  -->  Chroma analysis (Krumhansl-Schmuckler)  -->  Key + BPM lock
                                                         |
caption (per track) + reference audio  -->  ACE-Step DiT  -->  raw wav
                                                         |
raw wav  -->  seamless crossfade loop  -->  level normalization  -->  final output
```

The model uses a **cover task** rather than text-to-audio. The reference track is fed as a conditioning signal at controlled strength (`0.4`), anchoring harmony and tonal center across all generated tracks while leaving timbre and arrangement free.

## Requirements

- Python 3.10+
- CUDA GPU with 6-8 GB VRAM minimum
- ACE-Step 1.5 repo cloned locally

## Setup

```bash
# Clone and prepare ACE-Step
git clone https://github.com/ace-step/ACE-Step-1.5.git ~/dev/ACE-Step-1.5
cd ~/dev/ACE-Step-1.5
uv sync

# Install audio deps for this script
pip install soundfile torchaudio
```

## Usage

```bash
# From the ACE-Step repo root
cd ~/dev/ACE-Step-1.5
uv run python /path/to/generate_music.py
```

Models download automatically on first run. Output lands in `generated_tracks/`.

To regenerate specific tracks, edit `REGEN_PHASES` in the script. Set to `None` to regenerate all.

## Configuration

All tunables live at the top of `generate_music.py`:

- `MUSIC_TRACKS` — dict mapping phase names to caption, bpm, and crossfade settings
- `AUDIO_COVER_STRENGTH` — how much the reference melody shapes each track (0.0 to 1.0)
- `THEME_KEY` / `THEME_BPM` / `THEME_TIME_SIG` — global musical constraints
- `OUTPUT_LEVEL_DB` — peak normalization target
- `REGEN_PHASES` — subset of tracks to regenerate

## What's Novel

ACE-Step 1.5 uses a hybrid LM + DiT architecture where the language model acts as an omni-capable planner. It transforms simple prompts into structured song blueprints via chain-of-thought reasoning, then guides the diffusion transformer. Alignment is achieved through intrinsic reinforcement learning with no external reward models. This pipeline demonstrates the cover task workflow: conditioning a 2B DiT on reference audio at controlled strength to produce a multi-track soundtrack that shares harmonic DNA while varying instrumentation, tempo, and mood per game phase.

## Notes

Caption specificity matters. Including instrument names, tempo feel, mood keywords, and style references (Nintendo, JRPG, cyberpunk) produces significantly more accurate results than vague descriptions.

The LM planner (`LM_MODEL_SIZE`) is bypassed for cover tasks. The DiT alone handles generation when a reference audio is provided.
