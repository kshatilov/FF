# ff - fulcrum rig calibration CLI

`.\ff` is the entry point for FF (fulcrum-frustum) calibration. All
shared JSON configs live under `tools/pipe/ff/`; scripts and shaders
live here in `tools/ff/`; published docs (.tex/.pdf/.drawio) stay in
`tools/fulcrum/`. Diagnostics and archives go to `smoke/`.

```
.\ff calib                    # full rig: webcam + projector + fresh Kinect
                              #   color registration + SDK depth intrinsics
.\ff calib --no-kinect        # legacy webcam + projector only
.\ff calib -c_i 0             # one camera only: writes camera_c0.json
.\ff calib -c 1               # (-c, --camera, -c_1 all equivalent)
.\ff calib --restore          # restore TD bindings after a forced kill
```

## Camera indices

| id | device | record |
|----|--------|--------|
| `c0` | Logi webcam via `/project1/camLogi` | solved ff-marker pose, `camera_c0.json` |
| `c1` | Kinect color via `/project1/kinectC` | factory-mapper K + solved pose, `camera_c1.json` |
| `c2` | Kinect depth via `/project1/kinectD` | **SDK factory intrinsics only**, `pose_calibrated: false` — no faked RGB solve |

## Scope

- `.\ff calib` (whole rig) runs all legacy phases, then Kinect
  registration, then publishes `calibration_current.json`,
  `calib_geometry.json`, `td_kinect_registration.json`, and
  `camera_c0/1/2.json` **together** — staged; if any stage fails
  nothing is published and nonzero is returned (archives only).
- `-c_i N` writes **only** `camera_cN.json`. The joint rig record,
  projector, viewer geometry, and other camera records are untouched —
  a single-camera solve must not be mixed into the coupled rig.
- A failed or unvalidated run leaves the last good shared files in
  place.

## TouchDesigner output ownership

`OutputSession` snapshots `final_switch.index` (expr + value + mode),
`cam_and_kinect.index`, and `output_mode.Mode`, tags `/project1`
storage with the session owner, and restores every binding in
`finally` — including on errors and Ctrl+C. If the process is killed
before restore, run `.\ff calib --restore` to replay the snapshot and
clear ownership. Sessions refuse to start while an hg or FF session is
active.

## Shared store (`tools/pipe/ff/`)

Producers and consumers read/write these names only; the pre-move
copies under `tools/fulcrum/` and `tools/td/` are read-only backups.
`revisions/<uuid>/` keeps previous bytes on every publish for
recovery. The Node viewer `tools/td/calib_viewer.html` reads
`calib_geometry.json` from `pipe/ff` — reload the page after a new
calibration.

## Physical requirements

- The FF must be stationary and at least 3 of its physical markers
  (ids 10-14) visible to the camera during solves.
- Projector focus/keystone/zoom and all camera parameters must be held
  fixed; recalibrate after any change.
- Depth (c2) is not an RGB marker camera: its record is factory
  intrinsics in native SDK depth pixels, not a measured depth-to-FF
  pose.
