# FFv2 color-and-geometry tracker (prototype — NOT production ready)

Tracks the painted 5-vertex solid defined by the authoritative
`tools/pipe/ff/ffv2_spec.json` (base ABC edges 30 mm, DA=DB=DC=40 mm, E opposite B
across ADC, AE=CE=40 mm and DE=30 mm;
faces DBC black, ADB green, DEC red, ADE violet, AEC yellow, ABC ignored).
Detection = saturated-color face interiors + red-border geometry +
solvePnPGeneric hypotheses + bounded joint refinement (silhouette, red-edge
and face-color distance fields). No ArUco on this path.

## Sources (cam_and_kinect index)

- 0 kinectC color, intrinsics + webcam_from_kinect from td_kinect_registration.json
- 1 webcam 640x480, K from calibration_current.json
- 2 kinect depth: tracking uses the kinectC color companion
  (`source='kinect_depth+color'`); depth itself carries no color.

## TouchDesigner

Wiring: `cam_and_kinect -> ffv2_project -> ff_v2 -> final_switch[7]`,
`ffv2_debug <- cam_and_kinect`. `output_mode.Mode` menu entry `ffv2` is index 7.
Selecting it shows processed output in `ff_v2`.

**Projector output is gated OFF by default**: `ffv2_controls` toggle
'Experimental Projection' must be switched on to emit pixels. The node warns:
geometric projection accuracy is not yet validated. Debug overlay is
`ffv2_debug` (camera view, never the projector).

## Loss behavior

miss <=0.15 s: bounded prediction; then static hold of the coast pose fading
to black by 1.5 s; missing/malformed frames take the same path. Source switch
respawns the worker, clears the TOP and drops pending results.

## Status / limitations

- Small object size (20-50 px) makes pose estimation difficult.
- Latency also needs optimization: the final live snapshot recorded about
  550 ms of worker processing and 594 ms frame age. Earlier faster samples
  are not a reliable throughput guarantee.
- Strict synthetic quality bar is NOT met yet (see
  `smoke/ffv2_final_tests.log`): projector-vertex error and translation error
  exceed the 8 px / 30 mm limits — requires further work or a closer camera.
  Kinect-side detection is marginal and often honestly rejected.
