# FF — Fulcrum Frustum

*Shared Light: A Framework for Collaborative Projective XR*

FF is two things:

- **A conceptual framework**: a way to give many independent devices —
  cameras, projectors, phones, wearables, sensors — one shared spatial and computational
  medium, so an art piece or any content can be attached to a room or to moving objects.

- **A physical evidence**: the Fulcrum Frustum itself, a small
  measured, marker-covered truncated pyramid, a common ruler.
  Cameras observe it, and projectors are registered against it. Also acts a Statement of Work and a self-[`_doc/`](_doc/)umenting artifact. 


## The framework

![FF framework overview — four categories and the FF object at the center](FF_Overview.png)

The framework has **four categories**, shown as element colors in
the overview. These elements are just a creative organizational metaphor.

| Category | Element | Glyph | Role |
|----------|---------|-------|------|
| **SODI** | EARTH | 🟩 | Smart Objects / Dumb Interfaces — physical objects, embodied interfaces|
| **TD** |  LIGHT | 🟨 | TouchDesigner — projective mapping, animations, and sound production |
| **CBE** | WATER | 🟦 | Compute Backend — calibration, CV, tracking, services |
| **APP** | FIRE | 🟥 | Applications — experiences, demos, artworks |

Reading the diagram:

- A **diagonally split square** means an **intersection of categories**.

- **Arrows** mean enablier/enclosure.

- The **FF itself sits at the center and touches all four categories**:
  it is a physical artifact (SODI), a measured geometry, digital twin of itself (CBE), a bunch of photons projected on top of it (TD), and it is an interactive artifact (APP)/(DEMO).

## Project codes

| Code | Categories | Description |
|------|-----------|-------------|
| **FF** | CBE+TD+APP+SODI | The framework and its physical calibration reference. The framework includes a shared spatial language, neccessary algebra and guidelines for creating and documenting FFs. |
| **HW** | CBE+SODI | Hardware category |
| **SODI**| SODI/DOSI| Smart objects - dumb interfaces. Or dumb objects smart interfaces. Old idea of using regular non-instrumented objects as inputs for digital systems. Revibrates with MyoKey - holding a beer can and controlling smart glasses. Final delivaerable - Roman Dodecahedron (RDDH) - LED and wire-instrumented (yes) 3d-drawn tetrahedron.   
|**CBE**| CBE | Anything related to compute.
|**TD**| TD | Touch designer projects, integration scripts, the bridge.
| **SHELL** | CBE+TD | Shell / command interface (cli) for status checking, overseeing devices, calibrations, and workflows. |
| **RAI** | CBE | Remote AI: ComfyUI / LLMs on demand. |
| **LAI** | CBE | Local AI: MCPs, callbacks, and CLI bridges. |
| **API** | CBE | Integration and device orchestration scripts. Should be self-sufficient and not rely on agents. Examples are: blender 3D-sculpture -embedding scripts, touch designer reset-to-default shell scripts |
| **TDCP** | TD | TD Control Panel, config UI (Panel) sitting inside TD with previews and controls. Partly overlapping functionality with shell, compliments it with visual feedback. |
| **SI** | TD | Sound integration. Encapsultes audio library, dynamic soundscapes and curves for precise sound-to-digital event correspondence (e.g., pitch modulation based on collision distance of meta quest controller and digital boundary such as the donut). |
| **VJ** | TD+APP | Audio-reactive visuals. Prism Pulse exists and it is self-contained python ableton-bridge link with node.js frontend. Should be a library of cool TD presets with mic/system/pipe(ableton- implementation of it).  |
| **OART** | CBE | Objects and activity recognition and tracking. Complex, and computationally complex: (A) runtime and (B) a set of tools. Encapsulates everything related to: (1) digital capture (recording) of sensor (cam, depth, sEMG, IMU) data, (2) Pattern extraction and databasing the extracted patterns, and (3) Workers with access to the real-time feed of sensors who recognize the learnt patterns and report their change.  |
| **HMP** | CBE | Human mounted projectors [[(1)](https://youtu.be/Ls2Smn_-cbE?si=W2Aox2cy5kZyxKqg "AR Tabletop Interface using a Head-Mounted Projector"), [(2)](https://www.youtube.com/watch?v=IFI29N-dliw "Lightweight Wearable AR System using Head-Mounted Projector for Work Support")] hardware subproject. The general FF framework assumes moving cameras and _moving projectors_ as well. Challenges are hardware design, on-the-fly homography and spatial tracking. |
| **SB** | CBE | Smart bracelets, bands and rings (Myo+). Exciting oppotunity to control and read electrodes in smart wearable devices. Enables SODI. |
| **HCI** | SODI | Human-in-the-loop calibration and training. Protocols/checklists/instructions for calibration, and manual alignments (e.g. aligning a HMP's projector's beam with a shared landmark or a crosshair). |
| **VKC** | SODI | Virtual or ''disconnected'' keyboards (and controllers). Conceptually relies on sEMG recognition and visual hand tracking. An example - keyboard outlines in a notebook (can be hand-drawn) or a printed QWERTY layout. Something like [[3](https://arxiv.org/html/2410.20081v3 "emg2qwerty")]. Relies on OART sEMG decode and  OART Media Pipe hand tracking. Needs an OART-DT Double truth system, that captures 2 independent signals (tracked joint movements and electrical activity) and can supply ground truth (such as the ''disconnected'' keyboard can be actually connected and keystrokes are well-recorded with timespamps).  |
| **CHEBU** | SODI | Cheburashka plush toy as a controller subproject |
| **XRC** | TD+APP | XR Stadium subproject. Watching (e)sports together on a shared canvas. Old ideas, new life. |
| **XRF** | APP | XR Bonfire / shared collaborative projection. Relies on smoky environment (smoke machine or shisha) and multiple projective devices. |
| **XRA** | APP | XR art performances. |
| **XRT** | SODI+APP | XR Thesis. Tracking of a little black book, tracking hands via MediaPipe, projected visuals and texts complimenting the content of the book. |
| **DEMO** | APP | Asset generation and running demos. Includes scripts for blender integration, concat video editor and a gallery of produced artifacts |
| **XRBG** | SODI+APP | XR Board Games |


## Repository layout

```
ff
├── FF_Overview.png            # framework diagram
├── APP/
│   ├── DEMO/
│   │   ├── blender_integration/
│   │   └── concat_integration/
│   │   └── gallery/
│   └── XRT/
├── CBE/
│   ├── OART/
│   └── shell/
├── SODI/                      
└── TD/
|   └── connected_aligned.toe  # main TouchDesigner project (connecting inputs, TDCP v1, tracking ff v1, animations)
└── docs/                      
    └── shared_projective_xr.pdf  # main document
```
## Evidence vs. plan


| Status | Item | Where today |
|--------|------|-------------|
| Available code | ArUco-marked FF geometry spec; webcam + projector + Kinect calibration solver (`ff_calibrate.py`); live FF tracking prototypes | ProjectSynchronicity `tools/fulcrum/` |
| Available code | Isolated tracking worker and TD live-visualization modules (`ff_worker.py`, `td_ff_live.py`, `td_ffv2_live.py`) | ProjectSynchronicity `tools/td/` |
| Experimental | Myo armband → Web Bluetooth → TD input path (`myo/`) | same repo |
| Experimental | Audio-reactive browser/TD studies (Synchronicity app) | same repo |
| Experimental | Cheburashka anamorphic visual demo | same repo |



## Quick reference

Python environment:
```powershell
C:\Python314\python.exe (cv2/numpy/SciPy)
```

Calibration script:
```powershell
C:\Python314\python.exe ff_calibrate.py --help
```

Optional read-only bridge ping (requires the TD bridge already running
per `tools/td/README.md`):
```powershell
'{"op":"ping"}' | node tools/td/td.mjs --json -
```

Connect to projector
```powershell
adb connect 192.168.0.170:5555
```

Reset projector
```powershell
adb shell am force-stop com.fengmi.projector.imaging
```

Focus projector
```powershell
adb shell am start -a com.fengmi.projector.imaging.action.START_FOCUS
```

Switch to hdmi
```powershell
adb shell am start -a android.intent.action.VIEW -d "content://android.media.tv/passthrough/com.droidlogic.tvinput%2F.services.Hdmi1InputService%2FHW5" -n com.android.tv/.MainActivity
```

Make a screenshot from the proj embedded camera
```powershell
python tools/projector/projctl.py cam_shot .\cam_shot.jpg
```



## The Plan

| ID | Code(s) | Seed |
|----|---------|------|
| **OART-01** | CBE | Existing calibration pipeline cleanup|
| **TDCP-01** | TD | Fix and improve TD control panel |
| **SHELL-01** | SHELL | Hierarchical cli for controlling the projector and checking status of td-bridge |
| **XRT-01** | XRT | Measured specifications for the Thesis, aruco codes on pages of interest, some supplimentary materials |

TODO is [here](https://docs.google.com/spreadsheets/d/1zLj0124TzNesF9pex_uo_7AMlXkFmz83o-AXt5EHZVs/edit?usp=sharing "todo").


## Documents

- [`_doc/shared_projective_xr.pdf`](_doc/shared_projective_xr.pdf) —
  illustrated audited design.
- [`_doc/shared_projective_xr.tex`](_doc/shared_projective_xr.tex) —
  self-contained LaTeX source.


## Contributing

All are welcome.