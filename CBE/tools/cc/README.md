# cc - table color calibration

`cc` measures and registers a camera-relative table/projector white
balance on top of an `hg` geometry profile. The measurement loop and
the acceptance gate live in `tools/cc/measurement.py`; `cc` only
resolves profiles, stages inputs and publishes results.

```
.\cc -c 0 -p 0 -s_id Table_01            # measure, accept, publish
.\cc --load model.json -c 0 -p 0 -s_id Table_01
                                         # register a reviewed model
.\cc --list                              # enumerate shared profiles
```

Requires a geometry profile first (`hg -c <id> -p 0 -s_id <surface>`);
a missing one fails with that hint before any TD call. `c2` (depth) is
refused at argument validation - depth projection uses
`hg --map`/`--via` files, not an RGB color model.

## What it does

1. Resolves `tools/pipe/p0/<surface>/c<id>/profile.json` and
   writes its `P` doc verbatim as a private staging copy
   (`calibration_staging.json`) in a temporary directory - the
   measurement output directory itself (`-o`, default
   `out/cc/<surface>/c<id>/<utc>-<uuid>`) is created fresh by the
   measurement session.
2. Runs `measurement.calibrate(...)`: a TD session selects the camera
   on the shared `cam_and_kinect` switch and the surface on
   `hg_pair_controls`, projects gray ramps through the pattern gate
   (hg_output input 2), and reads frames back from the switch output.
   The acceptance gate in the measurement module decides - nothing is
   auto-approved.
3. On an accepted model only: `publish_color` attaches it to the
   profile (new `revision`/`photometric_revision`; `P`,
   `geometry_revision` and the hash are preserved verbatim, and every
   revision is archived under `revisions/`). Failed measurements or
   failed TD restores never replace the profile.
4. Sends the full validated bundle to TD over the wire (`register` -
   no TD filesystem reads). If the file is valid but registration
   fails, the artifact stays on disk and the command exits nonzero
   with `saved ... but TD load failed`.

## Normal operation

After `hg` published geometry and `cc` registered a model, the TD
frame watcher drives the corrected branch automatically: selecting
the camera on `cam_and_kinect` and the surface on `hg_pair_controls`
applies the matching profile; anything without a ready profile gates
black. `hg --warp` is only the initial-installation / manual
activation path, not a required third step.

The correction is measured attenuation: `gain_rgb * headroom`
scales the projector's drive so the camera sees neutral table gray.
It only ever reduces channel drive (projector black stays black,
headroom deliberately loses brightness because camera white clips).

`cc --load` performs steps 3-4 for an existing reviewed model.
Legacy models missing `destination_id`/`surface_id` may attach only to
the explicitly requested pair, recording migration provenance;
conflicting metadata is refused, never re-associated.

## Model and limits

`ff.hg.table_color` v1, `global_attenuation_gray_balance`:
`uTableGain = gain_rgb * headroom`, attenuation only. Headroom
deliberately loses brightness because camera white clips. It is not
colorimetric accuracy, ambient removal, spatial reflectance, or
optical feedback cancellation, and it assumes exposure, white
balance, geometry and ambient light stay fixed after capture.
TD-side controls: `hg_table_color_controls` `Enabled`/`Strength`
(user values preserved across reloads).
