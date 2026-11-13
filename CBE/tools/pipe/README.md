# tools/pipe — shared calibration store

This directory is the **data contract** between `hg` and `cc`. It is the
only place either tool reads or writes shared calibration artifacts.

## Canonical profiles

    tools/pipe/p0/<surface_id>/c<camera_id>/profile.json
    tools/pipe/p0/<surface_id>/c<camera_id>/revisions/<revision>/profile.json

`ff.projective.profile` v1: the homography document `P` plus the
accepted color model `TC` (or `null`), geometry hash, and revision ids.
Every publish archives the previous revision immutably before the
current file is atomically replaced. Surface ids are ASCII
`[A-Za-z0-9][A-Za-z0-9_-]{0,63}` — no traversal, reserved Windows names,
or case-insensitive duplicates.

    tools/pipe/p0/<surface_id>/c<camera_id>/P.json

Default standalone export of the `P` homography document (the `hg`
output path without `-o`; an explicit `-o` overrides the export only,
never the canonical profile).

## Maps

    tools/pipe/maps/c<a>_c<b>.json(.npz)     SDK camera<->camera maps
    tools/pipe/maps/c2_p0.json(.npz)         dense depth->projector maps

`hg --map` / `hg --via` artifacts.

## FF configs

    tools/pipe/ff/<name>.json

All shared Fulcrum-Frustum JSON configs live here:
`fulcrum_spec`, `calibration_current`, `td_kinect_registration`,
`ffv2_spec`, `control_params`, `fulcrum_obs`, `fulcrum_poses`,
`proj_state`, `td_node_positions`, `aruco_board`, `cheb_view`,
`aruco_solve`, `aruco_solve_frozen`, `calib_geometry`. Scripts that
produce or consume them default to this directory; the pre-move copies
under `tools/fulcrum/` and `tools/td/` are read-only backups.

## What does NOT live here

Raw measurement photos, logs, session reports and cc output directories
stay under `out/cc/` and `smoke/` — they are diagnostics, not the shared
store. Detection-failure frame dumps go to `smoke/hg_failures/`.

## Migration from out/calibration

The pre-pipe store `out/calibration/p0` is a read-only backup.
`hg --migrate-pipe` copies it here byte-exact (revision ids preserved),
idempotent, and refuses — never overwrites — a destination file with
different contents.
