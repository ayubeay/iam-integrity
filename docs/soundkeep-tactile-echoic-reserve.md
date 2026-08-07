# RESERVE - SoundKeep Tactile & Echoic Music Control Layer

**Status:** Reserved. Implement only after crates, Practice Decks, playback and Folder Sync
are stable.
**Urgency:** MEDIUM for Stage 1. The keyboard-complete path is worth building sooner than
the rest, and benefits every user.

## Objective
An accessibility-first interaction layer letting blind and visually impaired musicians
navigate, prepare and control music through touch, haptics, spatial audio and concise sonic
feedback rather than screens.

Begins as a software protocol and controller-mapping layer. Not hardware.

## Doctrine
Do not treat accessibility as a simplified version of the visual interface. Provide a
native interaction language: touch, physical landmarks, haptic patterns, echoic signals,
spatial audio, speech only when necessary, predictable control placement.

The goal is not to announce visual controls. It is to let a musician operate SoundKeep
without seeing the screen - which also serves sighted DJs who want to work without staring
at a laptop.

    SoundKeep music graph -> accessible interaction protocol
                          -> MIDI / keyboard / mobile haptics / future hardware

The tactile layer must not create a separate music database or duplicate DJ state.

## Phase 1 - controller-agnostic protocol
Every action carries an ID, input binding, current value, range, step behaviour, spoken
label, echoic cue, haptic cue where available, and a receipt event.

Actions: browse next/previous, load Deck A/B, play/pause, set cue, jump to cue, crossfader,
deck volume, tempo, select crate, start crate playback, announce current track, announce
next, announce position, request compatible transition, accept, reject.

## Phase 2 - five-control accessible mode
    1  browse / select      large stepped encoder, rotate to browse, press to select
    2  context / category   ridged encoder - Library, Crates, Decks, Search, Recommendations
    3  energy / filter      smooth knob or slider
    4  tempo / fine adjust  notched control
    5  confirm / load       large distinct button

Control identity must be exposed semantically so future hardware can use distinct shapes,
textures, heights, resistance, spacing and orientation landmarks.

## Echoic navigation
Restrained. Not a sound per action - brief recognisable motifs for state changes:
ascending two notes for a successful load, descending for unload, centred soft tone for
strong transition compatibility, dull tone for weak, short pulse for crate selected, double
for track loaded, rising and falling sweeps for next and previous, boundary tone at list
ends, warning tone for unavailable measurement or failed operation.

Users can disable, adjust volume, choose minimal or detailed, or use speech instead.
**Never route feedback through the master output during a performance.**

## Speech layer
Short and structured: "Test Crate. Four tracks." / "Track one of four." / "Up next: Mantra
Eight." / "Deck A loaded." / "Transition compatibility: high." / "BPM not measured."

Verbosity minimal, standard or detailed. The system must distinguish observed, unavailable,
unresolved and not applicable - **never announce an unavailable measurement as safe or
neutral.** That is the same doctrine the SURVIVOR holder-control work established.

## Haptic layer
One short pulse selection moved, two short confirmed, long warning, short-long unresolved
or unavailable, three short destructive requiring confirmation. Documented, consistent,
never mandatory.

## Spatial audio (reserve)
Current option centred, previous slightly left, next slightly right, Deck A left, Deck B
right, strong match centred. Optional; never interferes with the track being auditioned.

## Accessible crate workflow
    select DJ Crates -> hear crate names -> select -> hear track count and transition health
    -> browse in order -> hear position and up-next -> load first two into decks
    -> control playback, cue, volume, tempo, crossfader -> advance

Completable without a mouse.

## Receipts
receipt_id, timestamp, interaction_mode, input_device, action, previous and requested and
resulting state, target track / crate / deck, evidence_status, feedback_emitted,
haptic_pattern, speech_announcement, success, failure_reason, version. Local-first.

## Safety rules
Destructive actions require confirmation. Boundaries announced. Missing files never fail
silently. "Not measured" must not sound like "safe." Playback state always recoverable.
Controls predictable across sessions. Mappings exportable. No cloud dependency.

## Sequence
    Stage 1  software only - semantic action registry, keyboard-complete crate and deck
             navigation, concise announcements, focus management, accessible labels,
             position announcements, mapping config, receipts
    Stage 2  generic MIDI - Web MIDI or desktop, learn mode, saved mappings, profiles,
             five-control mode, echoic feedback
    Stage 3  desktop - native MIDI, separate cue output, persistent mappings, haptics
    Stage 4  hardware research - only after software validates the model

**Do not manufacture a controller before first-hand user testing.**

## Research requirement
Before promoting beyond reserve, speak with blind and visually impaired musicians,
producers, DJs, screen-reader users and accessibility specialists about actual workflows:
locating controls, changing modes, maintaining orientation, understanding state, avoiding
accidental actions, browsing large libraries, controlling two decks, receiving feedback
without masking music, recovering from errors.

## Guiding principle
A blind musician should not receive a spoken imitation of a visual interface. Offer a
native musical interface built from touch, position, rhythm, sound and predictable state.
The screen should be one interface to the music, not the only one.

## Grading
Stage 1 is worth building relatively soon - keyboard-complete navigation and clear state
announcements improve the product for everyone and cost little. Stages 2 to 4 wait for the
research, and the hardware waits for evidence that the interaction model works.
