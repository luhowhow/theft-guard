# Theft Guard

Real-time edge AI that catches break-ins at unmanned stores while they happen, on the computer the store already has.

Theft Guard reads the store's existing camera streams, finds people, follows them, and watches **what they do** in front of a kiosk. It does not recognize faces. When someone stays at a kiosk and keeps working on it, the system plays a voice warning on site, saves a photo every second, and sends a snapshot to the manager's phone. Video never leaves the store.

Entry for the ASUS UGen AI League, Lightning track (ASUS UGen300 USB accelerator). Stage I runs on the CPU of a 2020 entry-level office laptop; Stage II moves inference onto the UGen300.

- Demo video: [youtu.be/Xn8eUOB-638](https://youtu.be/Xn8eUOB-638)
- Slides: submitted with the Stage I proposal

## How it decides

| State | Box | Condition | Action |
| --- | --- | --- | --- |
| OK | Green | Outside every alert zone, or just passing by | None |
| WATCH | Yellow | In an alert zone for 5 s or more | Highlight only; the customer is not disturbed |
| ALERT | Red | WATCH, plus either 5 s of sustained limb motion, or 5 s of crouching with at least 2.5 s of motion | Voice warning, one photo per second, Discord message |

Alert zones are drawn with the mouse, any number per camera, and saved per camera (`cam1.yml`, `cam2.yml`, ...).

Two cues are tracked for each person inside a zone:

- **Motion**: the share of changed pixels inside the person's box. A frame counts as moving when it is above 30%; ALERT needs 5 s of such frames in total.
- **Crouching**: the box becomes shorter than 75% of the person's standing height. Crouching alone never triggers ALERT, because customers also crouch to pick up a parcel; it counts only together with at least 2.5 s of motion.

Motion is measured **body-aligned**: the previous frame is cropped at the person's previous box, resized to the current box, and compared. This separates arm movement from the person simply walking. Rules live in `src/camera.hpp` (`Rules`).

## Architecture

```
Camera streams (RTSP / file)
        │
        ▼
Decode threads ............. one per camera, keeps only the latest frame
        │
        ▼
Shared inference scheduler . YOLOX-tiny 416, cameras take turns
        │
        ▼
Per-camera pipeline ........ tracking, zones, behavior, OK / WATCH / ALERT
        │
        ▼
Actions .................... speaker, photos, Discord
```

- **Decode threads** (`src/frame_grabber.hpp`): one `std::thread` per camera, so a delayed or lost feed never holds up the others. Each keeps only the newest frame in a mutex-protected slot, so analysis always sees what is happening now.
- **Shared inference scheduler** (`src/guard.cpp`, `src/detector.hpp`): one YOLOX-tiny model analyzes one frame at a time with all CPU threads, cycling through the cameras. In Stage II only this part changes, to HailoRT on the UGen300.
- **Per-camera pipeline** (`src/tracker.hpp`, `src/zones.hpp`, `src/behavior.hpp`, `src/camera.hpp`): IoU tracker, foot-point zone test, dwell and motion timers, OK / WATCH / ALERT.
- **Actions** (`src/alert.hpp`, `scripts/notify.sh`): sound via `aplay`, photos to `events/camN/`, Discord webhook.

## Results

Test set: 27 self-recorded clips in 9 scenarios, 3 clips each, with the person approaching from a different direction in each clip. Only the author appears in them. The clips are kept out of this repository because of their size; download them from [Google Drive](https://drive.google.com/drive/folders/1oWozBmZL94ZQawxkJkuCc-obxQqD7nR_?usp=sharing) into `data/clips/`.

| Group | Scenarios (3 clips each) |
| --- | --- |
| Break-in attempts (6) | bare hand, hammer |
| Normal visits (21) | walk past, walk past carrying a tool, short stay, long stay, short crouch, long crouch, stay while handling a tool |

| Version | Thefts caught | False alarms |
| --- | --- | --- | --- |
| Raw pixel change, 2 s | 6 / 6 | 16 / 21 |
| Body-aligned motion, 2 s | 6 / 6 | 17 / 21 |
| **Body-aligned, 5 s (chosen)** | **6 / 6** | **6 / 21** |
| Body-aligned, 6 s | 5 / 6 | 4 / 21 |

Remaining false alarms are long crouches and customers handling a tool; a pose model is planned for Stage II. Per-clip results are in `bench/results.md`.

Performance on an Intel i5-1035G1 (4 cores, 8 threads, 15 W), CPU only, YOLOX-tiny 416:

- Inference is about 94% of the time per frame.
- One inference, pinned with `taskset`, each run after a 60 s cool-down: 1 core 139 ms; 2 cores 1.39×; 4 cores 1.62×; 8 threads 1.70×. A second Hyper-Thread on the same core adds only 16%, and the clock drops from 3.3 GHz (1 core) to 2.2 GHz (4 cores) under the 15 W limit.
- Running nonstop, the laptop manages 7–9 frames/s, enough for 1–2 cameras at 5 analyzed frames/s each.

| Configuration (pinned with `taskset`) | CPUs | Inference (ms) | Speedup | Avg clock |
| --- | --- | --- | --- | --- |
| 1 thread, 1 core | 0 | 139.2 | 1.00× | 3.3 GHz |
| 2 threads, 1 core (Hyper-Threading) | 0,4 | 120.4 | 1.16× | 3.2 GHz |
| 2 threads, 2 cores | 0,1 | 100.4 | 1.39× | 2.9 GHz |
| 4 threads, 4 cores | 0–3 | 85.7 | 1.62× | 2.2 GHz |
| 8 threads, all | 0–7 | 81.9 | 1.70× | 2.2 GHz |

Details and raw numbers: `bench/results.md`, `bench/threads_pinned.txt`.

## Build

Tested on Ubuntu with GCC, CMake and OpenCV 4.10 (DNN module, TBB backend).

```bash
git clone https://github.com/luhowhow/theft-guard.git
cd theft-guard
# Install dependencies and download the YOLOX models into models/
bash setup.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
# Video files: list as many as you have, one per camera
./build/guard data/clips/cam1.mp4 data/clips/cam2.mp4 data/clips/cam3.mp4

# A network camera
./build/guard rtsp://user:pass@192.168.1.20/stream1

# A webcam (device index)
./build/guard 0
```

Each argument is one camera, so pass as many as the store has. Sources can be mixed: a video file, an `rtsp://` URL, or a number for a local webcam. The order sets the camera number: the first argument is CAM 1 and uses `cam1.yml`, the second is CAM 2 and uses `cam2.yml`, and so on, so keep each camera in the same position every run.

Video files are replayed at their real frame rate, so they behave like live cameras.

**Drawing alert zones** (keys are also shown in the status bar):

| Key | Action |
| --- | --- |
| 1–9 | Choose which camera's zones you are editing |
| Mouse click | Add a corner; the outline closes automatically into a polygon |
| Enter | Finish the zone |
| s | Save zones to `camN.yml` |
| c | Clear the zones of this camera |
| q | Quit |

**Manager notifications:** copy `.env.example` to `.env` and fill in `DISCORD_WEBHOOK_URL`. `.env` is ignored by git and must never be committed.

## Evaluation and benchmarks

```bash
# Test clips: download from the Google Drive link in Results into data/clips/

# Current rules: per-clip table and summary
./build/eval data/clips/*.mp4

# Compare several rule settings
./build/eval data/clips/*.mp4 --sweep

# Time split: decode, pre-process, inference, post-process
./build/bench_single data/clips/x.mp4

# Thread scaling on pinned CPU sets, with clock sampling
bash scripts/pin_bench.sh
```

`eval` caches detections in `.eval_cache/`, so trying new rules takes seconds instead of re-running the model.

## Privacy by design

- No face recognition: the system judges behavior, not identity.
- All analysis runs inside the store; only event snapshots are sent to the manager.
- Photos are kept only for events, in `events/`, which is ignored by git.

## Repository layout

```
src/          guard (main program), eval, and the pipeline headers
bench/        benchmark programs and results
scripts/      notify.sh, pin_bench.sh, env_info.sh
experiments/  small step-by-step programs used during development
setup.sh      dependency installation
```

## Stage II plan

1. Compile YOLOX to Hailo's INT8 format and run it with HailoRT; measure cameras per UGen300, latency and power.
2. Larger detector on the NPU and ByteTrack for steadier IDs.
3. Pose model to separate prying from fidgeting.
4. Lookout pattern (one person at the kiosk, one at the door) and tool check with CLIP.
5. Evidence package: clip from before the alert with timestamps and SHA-256 hashes.
6. Larger, more varied test set.

## Third-party components

- YOLOX (Megvii), Apache 2.0
- OpenCV 4.10, Apache 2.0
