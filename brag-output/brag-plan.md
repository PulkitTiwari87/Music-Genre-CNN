# Brag Plan: Music Genre Classification (CNN V3)

## What is this app?
A convolutional network that reads Mel spectrograms of 3-second slices of a song and votes across them to name one of 10 genres: **90.0% of 150 songs it had never heard**, served as a web app.

## The angle
The project's own web film asks one question: *can a machine learn the structure of music?* The brag is that film in 23 seconds. What makes it specific: it shows the **real** spectrogram of a real held-out test song (`rock.00020.wav`), the **real** class probabilities the network gave it, the **real** ROC curves, and then admits where it fails. A model that says "15 of 150 were wrong" is more credible than a perfect one.

## Hook (first 2-3 seconds)
`MUSIC IS STRUCTURE.` over frequency lines that are driven by the actual Mel frames of that song, then `CAN A MACHINE LEARN IT?` landing on the beat.

## Key moments (the middle)
- The waveform of a real 30 s clip draws in, the first 3 seconds light up, and it turns into the real 128 × 130 Mel spectrogram (the exact model input).
- Four convolution blocks (32, 64, 128, 256 filters) light up one beat at a time; ten real probabilities rise; `ROCK 98.6%`.
- `90.0%` slams in on the strongest cue, then the real song-level ROC curves draw with `ROC-AUC 0.986` and `PR-AUC 0.946`.

## Outro / punchline
`15 OF 150 WERE WRONG. POP GETS LOST: 5 OF 15.` Then the title lockup `MUSIC GENRE CLASSIFICATION`, the real stack, and `Built, not just trained.`

## User flow worth showing
Entry → key action → result, as the app does it: **a song goes in → it becomes a spectrogram → the CNN answers with genre probabilities.** (The live demo at `/` uploads a file; the video uses the held-out test song so every number is checkable.)

## Tone
- Preset: cinematic
- Creative direction: a technical documentary about how a model learned to read music, honest about where it fails
- Interpretation: wide, dark, big type; dramatic reveals on strong beats; restraint on audio; the honesty beat is the emotional turn, not a joke.

## Format: landscape — 1920x1080
## Duration: 23 seconds

## Visual identity (from the project)
- Background: `#05040b` (the film's ink)
- Accent: `#8f78ff` (violet), secondary `#ffb45a` (warm, used for errors/honesty), `#fcfdbf` (cream, key numbers)
- Text: `#f4f2fc`, dim `#a09bbb`
- Display font: Bahnschrift (bundled locally; the site uses the system UI font stack)
- Body font: Consolas for technical labels
- Strongest visual element: the real magma-coloured Mel spectrogram and the genre-colour probability bars

## Share copy (draft)
Music Genre Classification. A CNN that reads spectrograms and named the genre of 90.0% of songs it had never heard, and shows exactly where it fails.

## Audio direction
- Role: cinematic support, restrained
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (about 110 BPM, steady, not saccharine)
- Music treatment: starts at 0 with a 0.6 s fade-in, volume about 0.5, fades out over the last second
- Music cue guidance: preset read (`vol-12`, 109.96 BPM). Strong cues used: 8.74 s (the "model input" tag), 13.11 s (the 90.0% slam), 18.56 s (the honest line). Beat-grid windows: 9.83 / 10.37 / 10.93 / 11.46 for the four conv blocks
- Audio-reactive treatment: subtle; bass energy breathes a violet glow behind the scene. No waveform or equalizer visuals (the spectrum lines in the hook are real Mel data, not a music analyser)
- SFX posture: sparse, motion-matched, low high-frequency risk
- Audio-coupled moments: four clicks on the four conv blocks; a bell on the verdict and on the title lockup; soft impacts on scene hits
- Restraint rule: nothing loud under a line the viewer is reading; no SFX on every beat

## Storyboard

### Scene 1 — Hook — 4.39s
Dark stage. `MUSIC IS STRUCTURE.` (0.15–1.9 s). Spectrum lines (real Mel frames of the sample) rise from a flat line. `CAN A MACHINE LEARN IT?` lands on the 2.19 s beat and holds to the cut.
Sequential/interaction: two lines in sequence; the spectrum level ramps with them.
Audio intent: a held breath, then a soft hit on the question.
Audio-coupled idea: soft impact on 2.19 s.
Music: bed fading in.
Transition mood: dramatic hard cut on the 4.39 s beat → Scene 2

### Scene 2 — Sound becomes a picture — 4.9s
Real waveform of the 30 s clip draws left to right; first 3 s highlight in warm with "first 3 s = one training example"; the waveform collapses and the real Mel spectrogram wipes in left to right with a scan line; `IT READS THE PICTURE.` lands; `128 × 130 × 1` tag locks on the 8.74 s strong cue.
Sequential/interaction: wipe reveal with scan line.
Audio intent: curiosity, then clarity.
Audio-coupled idea: card-slide whoosh as the spectrogram wipes; soft bong on the tag.
Transition mood: hard cut on 9.29 s beat → Scene 3

### Scene 3 — The model — 3.82s
Input spectrogram thumbnail on the left. Four slabs (32, 64, 128, 256 filters) light one per beat (9.83, 10.37, 10.93, 11.46). Ten real probabilities rise on the right, sorted. `ROCK 98.6%` slams on the 12.02 s beat.
Sequential/interaction: yes — slabs light one by one on consecutive beats; bars rise in a stagger. Slab labels are numbers only, so one per beat is readable.
Audio intent: mechanical build to a payoff.
Audio-coupled idea: click per slab; bell on the verdict.
Transition mood: hard cut on the 13.11 s strong cue → Scene 4

### Scene 4 — The evidence — 4.91s
`90.0%` slams on 13.11 s and counts up. `of unseen songs placed in the right genre`, `135 of 150`. At 15.29 s the number slides left and the real ROC curves (10 genres plus pooled) draw; `ROC-AUC 0.986` at 16.38 s; `PR-AUC 0.946` at 16.93 s. Both stay up until the cut.
Sequential/interaction: curve draw; two metric pills one after the other with full holds.
Audio intent: confidence, earned.
Audio-coupled idea: big soft impact on the slam; clicks on the pills.
Transition mood: hard cut on 18.02 s → Scene 5

### Scene 5 — Honest, then the title — 4.98s
`15 OF 150 WERE WRONG.` (18.02 s). `Pop gets lost: 5 of 15.` (18.56 s strong cue). On the 20.19 s beat: title lockup `MUSIC GENRE CLASSIFICATION` with the bell; the stack chips (TensorFlow · Keras · librosa · scikit-learn · FastAPI · React · TypeScript · Vite) at 20.75 s; `Built, not just trained.` at 21.28 s. Final fade to black over the last 0.5 s.
Sequential/interaction: text sequence, each held at least 1.2 s except the short one-word items.
Audio intent: a quiet admission, then the swell.
Audio-coupled idea: bell on the lockup; music fades out under the last line.
Transition mood: slow fade out.

**Music mood for this video:** cinematic-leaning upbeat bed, kept low
**Audio summary:** the bed fades in under a held breath, builds through four clicks to a bell on the verdict, hits big on 90.0%, goes quiet for the honest line, rings once on the title and fades.
