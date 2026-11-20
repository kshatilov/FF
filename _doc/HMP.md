# HMP — Human-Mounted Projectors

*The hardware heart of the Shared Light Project. You carry a light, you control the light, and the space does not forget where the light goes.*

## TL;DR

Shared Light (the **FF** project) lets a plethora of independent devices — cameras,
projectors, phones, sensors — draw into **one shared spatio-digital (phygital) canvas**. The part that
makes it fun to build is the **HMP** (Human-mounted projector): a projector you _hold_, _wear_ or _park_.

Under the hood it is a moving light that *knows where it is* and *knows where some others are*, in a space that contains known spatial markers (such as **FF** itself) and some static cameras and projectors.

Main challenges are:

- Wiring and driving DLP modules;
- Wiring and getting real-time feed from Sensors (IMUs) and camera modules;
- Designing and implementing lightweight on-device compute boards, something like Sparthan board that has minimal delay reads from IMUs and drives the envisioned on-device gimballs (stabilizers). 
- Designing and connecting communication boards, such as WiFi and type-C port communications for firmware flashing, debug, and communication with a phone or PC. 
- Supplying power to the above components.  

---

## The key idea

Three questions are on the table:

- **What exists** — an artwork (a virtual sculpture) or any digital (mutable, as can be draw in real-time) content to display.

- **Where it is** — where it is attached to. Can be the geo marker (monument, building, Victoria Harbour), a wall in the alley, or a simple object (such as FF, QR code on the paving/ground).

- **Who looks at it** — each consumer (viewer/observer) and each producer (the artist shaping the sculpture). Their perspective (world coordinates, focal vector and inner state) matters. 

---

## Meet the HMP

Below are the conceptual illustrations (GPT generated), introducing the concept of HMP. 

<table>
  <tr>
    <td align="center">
      <img src="img/1.%20Glasses-mounted.png" height="300" alt="Glasses-mounted"><br>
      <sub>Glasses-mounted projector</sub>
    </td>
    <td align="center">
      <img src="img/2.%20Headphones-mounted.png" height="300" alt="Headphones-mounted"><br>
      <sub>Headphones-mounted  projector</sub>
    </td>
  </tr>

  <tr>
    <td colspan="2"><br><br></td>
  </tr>

  <tr>
    <td align="center">
      <img src="img/3.%20The%20lantern.png" height="300" alt="The lantern"><br>
      <sub>The lantern</sub>
    </td>
    <td align="center">
      <img src="img/4.%20Phone-mounted.png" height="300" alt="Phone-mounted"><br>
      <sub>Phone-mounted projector</sub>
    </td>
  </tr>
</table>

All of the designs share the following principles:

1. **Camera and projector are 'glued' to each other.** Their mutual offset is constant, however the homography (correspondence between pixels of the camera and the projector) changes (TODO - given the fixed mount and variable projection surface should be some low-effort recalculation). 



---

## Quick reference for the compute framework

Same names, now as a transform chain. (Read right-to-left: each arrow maps a point out
of one frame into the next.)

```
     F   fulcrum (the ruler, see below)
     │
     │   T_W←F      known / tracked
     ▼
     W   world (the room)
     │
     │   T_W←H      Route B: track the housing
     ▼
     H   housing      ← compute + battery + handle (the heavy part)
     │
     │   T_H←B(α)    Route B: gimbal joint angles (encoders)
     ▼
     B   carrier      ← camera + projector (the part you aim)
    ╱ ╲
   C   P              ← fixed mounts: T_B←C and T_B←P never change
```

Two valid routes, and the project is explicit that **both are optional**:

- **Route A — track the carrier directly.** The camera rides on the carrier and sees
  the fulcrum. You get `T_W←B` straight away, **no encoders required**. Encoders stay
  useful for control, limits and diagnostics, but they are not on the critical path.
- **Route B — track the housing and measure the articulation.** A world-aligned tracker
  gives `T_W←H`; synchronized measured joint angles give `T_H←B(α)`. Chain them.

The honest catch that hardware people will enjoy: a motor *command*, a "stabilised"
indicator, or a housing IMU is **not** a pose. Motor lag, backlash, joint deflection and
encoder errors all go into the uncertainty budget. And a gimbal with nonzero lever arms
changes *position as well as pointing* when a joint rotates — it is not orientation-only.

---

## The ruler: the Fulcrum Frustum (FF)

Every device needs a shared ruler. This is it — a small, measured, marker-covered
truncated pyramid. Cameras observe it; projectors are registered against it. Print,
fold, done.

```
        ▁▁▁▁▁▁▁▁▁▁▁▁   top 75 mm        [id 10]
       ╱            ╲
      ╱   side       ╲   side markers 50×50 mm
     ╱    markers     ╲  [id 11 · 12 · 13 · 14]
    ╱_________________╲
        base 155 mm
    height ≈ 42.5 mm    (face tilt ≈ 46.7°, not 45°)
```

Small, cheap, and unforgiving in the best way: it has **measured** dimensions, not the
nominal ones you meant to print, so the whole system is anchored to reality instead of
a convenient ideal pyramid. A projector is treated as an *inverse camera* — it cannot
see markers, so you shine known patterns and let a camera tell you where the light
actually landed.

---

## The rest of the project (so you know where HMP sits)

Four categories, named as elements, colour-coded. The FF object sits in the middle and
touches all four:

| Category | Element | Colour | What it owns |
|----------|---------|--------|--------------|
| **SODI** | Earth | green  | physical objects, embodied interfaces |
| **TD**   | Light | yellow | TouchDesigner — mapping, animation, sound |
| **CBE**  | Water | blue   | compute, CV, tracking, calibration |
| **APP**  | Fire  | red    | experiences, demos, artworks |

**HMP** lives in **CBE** (with a hand in SODI for the housing): the framework assumes
*moving cameras and moving projectors*, not just fixed rigs.

---

## What already exists vs. what's next

Evidence today (real code, real hardware):

- ArUco-marked fulcrum geometry spec, plus a webcam + projector + Kinect calibration
  solver and live tracking prototypes.
- A Fengmi projector driven over ADB (autofocus, keystone, HDMI switch, embedded-camera
  screenshot).
- Myo armband → Web Bluetooth → TouchDesigner input path (EMG/IMU/poses).

The hardware backlog, in honest gate order:

- **H1** Survey and label the reference geometry (measured, versioned).
- **H2** Build the rigid optical carrier on its common gimbal — and prove the
  camera↔projector offset stays invariant across the full joint sweep.
- **H3** Verify electrical and thermal budgets — PD profiles, startup load, motor
  idle/slew/dynamic load, fuses, sustained runtime, no brownouts.
- **H4** Optical and mechanical safety — IEC 62471-5 assessment, emergency stop,
  pinch-point guarding, cable/thermal protection. A handheld beam sweeps bystanders.
- **H5** Package the portable lantern — carry/park/sweep/pack must **preserve** the
  carrier calibration.

Each gate has a concrete "done" that requires saved evidence, not a plausible
screenshot. That is the part of this project that will make a hardware person grin.

---

## Why you should care

Because somebody has to build the thing that turns "an art piece attached to a room"
into a device you can hand to a performer. It has a real gimbal to design, a rigid
optical carrier to machine, thermal and electrical budgets to pass, an eye-safety
standard to respect, and a calibration discipline that treats encoder angle and
projector pose as two different quantities.

The software side already knows the algebra. The fun part — the part with backlash,
lever arms, service loops and a transport lock — is waiting for hardware hands.

*Come build the lantern.*