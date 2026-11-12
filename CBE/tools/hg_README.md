# hg - homography / device-map utility

`hg` produces planar correspondences between the project's logical
devices and writes them as self-describing JSON artifacts. It is a
local tool - there is no installer and nothing to add to PATH. All math
and diagnostics come from `tools/hg_geometry.py`.

All live frames arrive through **TouchDesigner**: camera reads come
from existing TOP nodes, depth comes from the hg-owned
`hg_depth_xyz` point-cloud helper, and the calibration pattern is
projected through the existing `OART -> hg -> final_switch input 7`
branch. There is no OpenCV device capture, no desktop fullscreen
window, and no ADB traffic.

```
.\hg --help                      # no dependencies, no output files
.\hg --list                      # logical devices + TD branch state
.\hg --setup-td                  # create helpers, wire input 7 <- hg
.\hg --restore-td                # recover a session after a killed run
```

From PowerShell use `.\hg` (the repo-root `hg.cmd` wrapper); it calls
`C:\Python314\python.exe tools\hg.py` unless `HG_PYTHON` is set.
Equivalent direct form:

```
C:\Python314\python.exe tools\hg.py --list
```

## Logical devices

| id | device | space |
|----|--------|-------|
| `c0` | webcam via TD `/project1/camLogi` (VideoDeviceInTOP) | actual TD raster (640x480 on the current box) |
| `c1` | Kinect color via TD `/project1/kinectC` | TD raster 1280x720 (native SDK space is 1920x1080) |
| `c2` | Kinect depth via TD `/project1/hg_depth_xyz` (KinectTOP `depthpointcloud`, metres XYZ float RGBA, converted to uint16 mm) | 512x424 |
| `p0` | Fengmi M055MGN projector | actual TD branch raster (`/project1/OART`, e.g. 1280x720) |

Note: `/project1/kinectD` is an **infrared** image and is never touched.
`p0`'s ADB address `192.168.0.170:5555` is recorded only - `hg` makes no
ADB calls. `-p` is the *logical projector index*, not a monitor index.
The camera feeds are raw TD source nodes; any TD-side crop/flip/resize
would change the measured pixel space (and `--via` refuses unknown
rasters for that reason).

## Projector setup

Set the projector to the HDMI input and keep focus, keystone, zoom and
any auto-correction fixed; recalibrate after any of these change.

## One-time setup (`--setup-td`)

Creates hg-owned helpers only - idempotent, refuses to reconfigure
nodes it does not own, and never changes the output selection:

- `hg_pattern` Script TOP + `hg_pattern_callbacks` Text DAT (serves the
  float RGBA pattern array held in TOP storage via `copyNumpyArray`;
  outputs black until a pattern is pushed).
- `hg_depth_xyz` KinectTOP (`image='depthpointcloud'`,
  `mirrorimage=False`) if absent.
- Connects `final_switch` input 7 <- `/project1/hg` (the desired
  persistent wiring). The switch `index` expression and
  `output_mode` controller are not touched.

Setup runs automatically before each live command; `--setup-td` is also
available to install or inspect the branch explicitly.
Setup runs a read-only preflight of the whole
branch (node types, `kinectC` color/unmirrored, `hg` input exactly
OART, no feedback cycle, input 7 available, no active session) and
refuses to mutate anything on mismatch.

## Camera -> projector (c0/c1 -> p0)

`hg` briefly rewires `hg_pattern -> OART -> hg` and sets
`final_switch.index` to a constant 7 **only while a live calibration
runs** (an explicitly user-run command). The exact prior input,
parameter value, expression, and mode are snapshotted first and
restored in a `finally` block; a killed run can be recovered with
`.\hg --restore-td`.

All runtime images travel through the websocket as `hg_wire`
zlib-base64 arrays - the pattern is pushed into `hg_pattern` TOP
storage, and camera/depth frames are returned as arrays. No pattern,
PNG or depth files are staged on disk, which removes the reported
`pattern_path` failure point entirely (the underlying cause of that
report was never identified). Only the helper source files
(`tools/td/hg_td.py`, `tools/hg_wire.py`) must remain reachable by TD
at their repo paths. If TD restore fails the session snapshot is
retained in TD storage and the run exits nonzero without publishing an
artifact - fix the cause and run `.\hg --restore-td`.

```
.\hg -c 0 -p 0                 # webcam sees the ArUco grid on HDMI
.\hg -c 1 -p 0                 # Kinect color instead
```

The grid is uniquely-IDed ArUco (DICT_4X4_250, ids 100-111,
4 columns x 3 rows - no chessboard 180-degree ambiguity). Capture
retries for 10 s after a 2 s settle; Ctrl+C cancels and still restores.
At least 4 marker IDs (16 corners) must be detected. The first frame
meeting the marker count wins.

Default output is `out/hg/c0_p0.json` / `out/hg/c1_p0.json`. Override
with `-o`, label the measured plane with `--plane`:

```
.\hg -c 0 -p 0 --plane lab_wall -o out\hg\desk_run1.json
```

## Projected camera view (`--warp`)

After calibrating, route the raw camera through a GLSL inverse-UV pull
so the projector draws the camera's view of the measured plane:

```
.\hg --warp -c 0 -p 0            # uses out\hg\c0_p0.json
.\hg --warp -c 1 -p 0 --calibration out\hg\c1_p0.json
```

Route: `/project1/camLogi -> hg_warp (GLSL) -> OART -> hg ->
final_switch[7]` (index is pinned to 7 while the warp is active).
The shader (`tools/td/hg_warp.frag`) samples the raw camera raster at
`H_destination_uv_to_source_uv` and outputs black outside the mapped
region. All of it runs on the GPU every frame - no per-frame CPU work.
A later recalibration of the same pair reloads the coefficients
automatically without touching routing or selection.

`transformCam2` sits BEFORE OART and was bypassed during calibration
(the pattern was injected directly into OART and the camera read was
raw `camLogi`). The warp therefore consumes the raw raster too - a
physical upside-down camera is already absorbed into H, so do NOT add
a 180-degree rotation after the warp; that would double the flip and
break alignment. `transformCam2` itself is left fully intact (all
parameters, its own input) - only OART's input is replaced.

Limits: the warp is the same single-plane, raw-distorted-pixel planar
approximation as the calibration - off-plane geometry will not line
up. Re-projecting the framebuffer's RGB can produce feedback
brightening / hall-of-mirrors; this is a geometric mapping, not
reflectance capture or photometric cancellation.

## Offline correspondence mode (`--points`)

Fit a camera-to-projector homography from measured point pairs with no
TD, hardware, monitor, or SDK access (still needs `cv2`/`numpy`):

```
.\hg -c 0 -p 0 --points pairs.json --plane lab_wall -o out\hg\run.json
```

`pairs.json` schema (this exact file is a **synthetic** example -
uP = 3*uC + 20, vP = 2*vC + 30 - not a calibration):

```json
{
  "camera_size": [640, 480],
  "projector_size": [1920, 1080],
  "camera_points": [[100, 100], [500, 110], [490, 380],
                    [110, 370], [300, 240], [200, 300]],
  "projector_points": [[320, 230], [1520, 250], [1490, 790],
                       [350, 770], [920, 510], [620, 630]]
}
```

Coordinates are pixel centres; every point must lie inside its image
bounds.

## Kinect SDK maps (`--map`)

Dense forward pixel lookups through the native `CoordinateMapper`,
sanitized by `hg_geometry.sanitize_sdk_map`. Depth frames are captured
through TD `hg_depth_xyz` (native 512x424 point cloud); only the SDK
mapper handle is opened locally - no stream readers, and TD is never
restarted or saved.

```
.\hg -c 1 -c 2 --map      # color -> depth  (out/hg/c1_c2.json)
.\hg -c 2 -c 1 --map      # depth -> color  (out/hg/c2_c1.json)
```

Emitted coordinates are **native SDK** spaces (color 1920x1080, depth
512x424) - different from the 1280x720 TD color raster, so consumers
must apply explicit pixel-centre rescaling. Output is a JSON artifact
plus a sibling `.npz` (`xy` float32 `[srcH, srcW, 2]` with `[-1,-1]`
sentinels, `valid` uint8, `depth_mm` uint16 424x512). These are
*snapshots* (`snapshot_only`, `depth_required`) labelled
`native_sdk_pixels` / `factory_sdk_coordinate_mapper` - forward maps
only, never feed them to `cv2.remap`; run the reverse `--map` for pull
sampling. The `valid` mask means finite, in-bounds, nonzero depth only -
no visibility, occlusion, or confidence guarantee, and no camera or
projector rays are modelled. `--map` never touches the projector
selection.

## Depth -> projector (`-c 2 -p 0 --via`)

Depth can't see a projected pattern, so compose through an existing
color calibration:

```
.\hg -c 1 -p 0                        # first: live c1 -> p0 homography
.\hg -c 2 -p 0 --via out\hg\c1_p0.json   # -> out/hg/c2_p0.json
```

Without `--via` the command fails with instructions. The via artifact
must be a raw-pixel c1->P0 homography at a known 16:9 raster
(1920x1080 native or 1280x720 TD); other crops/mirrors are refused
before any hardware opens. A TD-raster source is converted to native
SDK space with `hg_geometry.resized_source_homography`
(full-frame pixel-centre resize assumption, recorded in provenance).
The output stores the dense depth->projector snapshot plus frozen
copies of both the original and converted homography, the coordinate
convention, space/policy labels, and the provenance path. Only points
on the via calibration plane are geometrically registered - observed
depth elsewhere does not turn the homography into a 3D warp.

## Output contract (`schema: ff.hg`, version 1)

- `kind`: `homography` | `dense_map`; `created_utc` ISO-8601.
- `source` / `destination`: `{id, name, type, resolution}` plus
  provenance - live sources record `capture_backend: "touchdesigner"`
  and `top_path`; live destinations record the `td_branch`
  (`hg` TOP, `final_switch` input 7, OART raster source) instead of a
  desktop monitor.
- Pixel coordinates: top-left origin, x right, y down, integer pixel
  centres, column vectors - native device space, no mirroring or TD
  convention.
- Homographies: `H_source_to_destination`, `H_destination_to_source`,
  plus `H_source_uv_to_destination_uv` and inverse built with
  `hg_geometry.uv_homography` (UV = bottom-left, half-pixel centres -
  the TD texture convention).
- Homography artifacts carry `source_space: "raw_distorted_pixels"`
  and `lens_policy: "planar_approximation_no_distortion_correction"`.
  **No lens calibration is applied.** Results hold only for the
  measured plane and fixed device configuration; `diagnostics` report
  RANSAC fit residuals (`independent_validation: false`), never a
  verified-accuracy claim.
- JSON is written atomically (unique temp sibling + `os.replace`,
  `allow_nan=False`); dense `.npz` payloads get unique versioned names
  written completely *before* the JSON, so metadata never points at a
  half-written map. Existing outputs are never deleted, and a failed
  run leaves any previous JSON untouched.
- Loading a dense map: `np.load(npz)` gives `xy`, `valid`,
  `depth_mm`; honour `valid` / `[-1,-1]` sentinels and
  `map_direction`.

## Coordinate spaces and TD

Rescaling raw pixels between rasters (e.g. native 1920x1080 <->
TD 1280x720):

```
x' = (x + 0.5) * newW / oldW - 0.5     # pixel-centre convention
y' = (y + 0.5) * newH / oldH - 0.5
```

Artifacts are meant to be read in TouchDesigner with a File In DAT
followed by `json.loads` - `hg` makes no TD network or auto-import
changes beyond the documented `hg_pattern` / `hg_pattern_callbacks` /
`hg_depth_xyz` helpers and the input-7 wiring.

## Requirements

- `C:\Python314\python.exe` with `cv2` (aruco) and `numpy`, plus
  `node` on PATH for `tools/td/td.mjs` (TD_WS_URL honoured).
- Live paths need TouchDesigner running the connected_aligned project
  with the wscontrol bridge on :9980.
- `--map` and `--via` additionally need the Kinect v2 SDK/drivers,
  `comtypes`, and the in-repo `tools/kinect/pykinect2` (mapper handle
  only - all frames come from TD).
- `--help` needs nothing; `--list`/`--points` work without TD or SDK.
