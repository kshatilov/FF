from pathlib import Path

root = op('/project1')
created = set()


def ensure(node_type, name):
    node = root.op(name)
    if node is None:
        node = root.create(node_type, name)
        created.add(name)
    return node


logic = ensure(textDAT, 'ffv2_logic')
try:
    stop = getattr(logic.module, 'shutdown', None)
except Exception:
    stop = None
if stop:
    stop()
logic.text = Path('C:/dev/ProjectSynchronicity/tools/ff/td_ffv2_live.py').read_text()

callbacks = ensure(textDAT, 'ffv2_callbacks')
callbacks.text = ("def onCook(scriptOp):\n"
                  "    op('/project1/ffv2_logic').module.cook(scriptOp)\n")
project = ensure(scriptTOP, 'ffv2_project')
project.par.callbacks = callbacks.path

debug_callbacks = ensure(textDAT, 'ffv2_debug_callbacks')
debug_callbacks.text = ("def onCook(scriptOp):\n"
                        "    op('/project1/ffv2_logic').module.cook_debug(scriptOp)\n")
debug = ensure(scriptTOP, 'ffv2_debug')
debug.par.callbacks = debug_callbacks.path

ff_v2 = root.op('ff_v2')
cam = root.op('cam_and_kinect')
switch = root.op('final_switch')

# Wiring seam: ff_v2 stays the user-facing processed OUTPUT at index 7.
#   cam_and_kinect -> ffv2_project -> ff_v2 -> final_switch[7]
#   ffv2_debug watches cam_and_kinect (the raw selected source).
# Disconnect ff_v2's old input first so no cycle is created.
for connector in ff_v2.inputConnectors:
    for c in list(connector.connections):
        c.disconnect()
project.inputConnectors[0].connect(cam)
ff_v2.inputConnectors[0].connect(project)
debug.inputConnectors[0].connect(cam)
switch.inputConnectors[7].connect(ff_v2)
switch.par.index.expr = "op('/project1/output_mode').par.Mode.menuIndex"
switch.par.index.mode = ParMode.EXPRESSION

# experimental opt-in control: created once with the toggle OFF; a rerun
# must never reset a user change
controls = ensure(baseCOMP, 'ffv2_controls')
if getattr(controls.par, 'Experimentalprojection', None) is None:
    # default 0 only on first creation — never reset a user's choice later
    page = controls.appendCustomPage('Tracking')
    page.appendToggle('Experimentalprojection', label='Experimental Projection')
    controls.par.Experimentalprojection = 0
    controls.comment = ('Prototype: geometric projection accuracy not yet '
                        'validated. Enable only to test.')

# isolated executeDAT: only cooks FFv2 when ff_v2 is the selected output
tick = ensure(executeDAT, 'ffv2_tick')
tick.text = ("def onFrameStart(frame):\n"
             "    sw = op('/project1/final_switch')\n"
             "    i = max(0, min(int(sw.par.index.eval()), len(sw.inputs)-1))\n"
             "    if sw.inputs and sw.inputs[i].name == 'ff_v2':\n"
             "        op('/project1/ffv2_project').cook(force=True)\n"
             "        op('/project1/ffv2_debug').cook(force=True)\n")
tick.par.framestart = True

# output_mode menu: append ffv2 entries, preserving all existing names/labels
mode = root.op('output_mode').par.Mode
names, labels = list(mode.menuNames), list(mode.menuLabels)
if 'ffv2' not in names:
    names.append('ffv2')
    labels.append('FFv2 track')
    mode.menuNames = names
    mode.menuLabels = labels
elif list(mode.menuNames) != names:
    pass  # existing entries untouched

# Positions only for nodes created in this run — manual arrangements preserved.
anchor = ff_v2 if ff_v2 is not None else project
for node, dx, dy in ((project, 260, 0), (debug, 520, 0), (logic, 260, -220),
                     (callbacks, 400, -220), (debug_callbacks, 560, -220),
                     (tick, 700, -220), (controls, 700, 0)):
    if node.name in created:
        node.nodeX = anchor.nodeX+dx
        node.nodeY = anchor.nodeY+dy

project.cook(force=True)
