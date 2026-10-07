# Hyperframes Composition Brief: Music Genre Classification

## Objective
Create a short cinematic launch-style brag video for the Music Genre Classification project (CNN V3).

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 23 seconds

## Source Material
- Project root: `D:\Claude Code\Music-Genre-CNN`
- Primary files read: `frontend/src/film/*` (the web film), `frontend/src/data/notebook.json`, `model.json`, `frontend/public/analytics/eval.json`, `README.md`
- Product name: Music Genre Classification
- Tagline / strongest claim: "Can a machine learn the structure of music?" / 90.0% of unseen songs placed in the right genre
- Key UI or visual moment to recreate: the real Mel spectrogram of `rock.00020.wav`, the real class-probability bars, the real song-level ROC curves
- Copy that must appear verbatim:
  - MUSIC IS STRUCTURE.
  - CAN A MACHINE LEARN IT?
  - 90.0% / of unseen songs placed in the right genre
  - ROC-AUC 0.986 · PR-AUC 0.946
  - 15 OF 150 WERE WRONG.
  - MUSIC GENRE CLASSIFICATION
  - Built, not just trained.
- All numbers are in `assets/film-data.js` (generated from the project files; do not retype them)

## Creative Direction
- Tone preset: cinematic
- Creative direction: a technical documentary about how a model learned to read music, honest about where it fails
- Interpretation: big type, wide dark stage, dramatic cuts on strong beats, restrained audio, an honest turn near the end
- Angle: the web film's question, answered with real data in 23 seconds, ending on the failures
- Hook: MUSIC IS STRUCTURE. → CAN A MACHINE LEARN IT? over spectrum lines drawn from real Mel frames
- Outro / punchline: 15 OF 150 WERE WRONG. Pop gets lost: 5 of 15. Then the title lockup, the stack, "Built, not just trained."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals; every visual is project data
  - Mood or wavelet claims (the repo has neither)
  - Equalizer / waveform-visualizer graphics driven by the music

## Visual Identity
- Background: `#05040b`
- Text: `#f4f2fc` (dim `#a09bbb`)
- Accent: `#8f78ff`; warm `#ffb45a`; cream `#fcfdbf`
- Display font: Bahnschrift (`assets/fonts/bahnschrift.ttf`)
- Body font: Consolas (`assets/fonts/consola.ttf`)
- Visual references from the project: magma-coloured spectrogram, genre-colour bars (blues #5b9cff … rock #ef4444), the violet/amber pairing of the web film

## Storyboard
Use the storyboard in `brag-plan.md` as the creative contract.

Scene summary:
1. Hook — 4.39s — two lines over real spectrum lines
2. Sound becomes a picture — 4.9s — real waveform → real spectrogram, 128 × 130 × 1
3. The model — 3.82s — four conv slabs on four beats, real probabilities, ROCK 98.6%
4. The evidence — 4.91s — 90.0%, then real ROC curves with AUC and PR-AUC
5. Honest, then the title — 4.98s — 15 of 150 wrong, pop gets lost, title lockup and stack

## Audio
- Audio role: cinematic support, restrained
- Audio arc: fade in → soft hits → four clicks → bell → big soft impact on 90.0% → quiet admission → bell → fade out
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
- Music treatment: 0.6 s fade-in, about 0.5 volume, fade out over the last second
- Music cue guidance: bundled preset `vol-12` (109.96 BPM): strong cues 8.74, 13.11, 18.56; beats 9.83, 10.37, 10.93, 11.46 for the conv slabs
- Audio-reactive treatment: subtle violet glow behind the scene from bass energy (`assets/audio-data.js`, 8 bands at 30 fps); no visualiser graphics
- Audio-coupled moments:
  - Scene 3 conv slabs — one click per beat
  - 12.02 s ROCK verdict — bell
  - 13.11 s 90.0% — big soft impact
  - 20.19 s title lockup — bell
- SFX selection guidance: low high-frequency-risk files from `sfx-analysis.md` (impactSoft_medium, impactBell_heavy, click_003, bong_001, card-slide-1)
- Exact SFX choice: chosen against the implemented animation (see `index.html`)
- Audio files: copied to `composition/assets/`

## Hyperframes Instructions
Use `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`, `hyperframes-cli`. Single paused GSAP timeline, no render-time randomness or clocks, local assets only, `npx hyperframes check` as the gate before render.
