from pathlib import Path

root = op('/project1')
created = set()
def ensure(node_type, name):
    node = root.op(name)
    if node is None:
        node = root.create(node_type, name)
        created.add(name)
    return node

source = Path('C:/dev/ProjectSynchronicity/tools/ff/td_ff_live.py').read_text()
logic = ensure(textDAT,'ff_live_kinect_logic')
try:
    stop = getattr(logic.module,'shutdown',None)
except Exception:
    stop = None
if stop:
    stop()
logic.text = source + '\nconfigure_kinect()\n'
live = ensure(scriptTOP,'ff_live_kinect')
callbacks = root.op('ff_live_kinect_callbacks') or live.par.callbacks.eval()
if callbacks is None:
    callbacks = root.create(textDAT,'ff_live_kinect_callbacks')
    created.add('ff_live_kinect_callbacks')
callbacks.name = 'ff_live_kinect_callbacks'
callbacks.text = "def onCook(scriptOp):\n    op('/project1/ff_live_kinect_logic').module.cook(scriptOp)\n"
live.par.callbacks = callbacks.path
resize = ensure(resolutionTOP,'ff_tracking_resize')
resize.inputConnectors[0].connect(root.op('kinectC'))
resize.par.outputresolution = 'custom'
resize.par.resolutionw,resize.par.resolutionh = 960,540
gray = ensure(monochromeTOP,'ff_tracking_gray')
gray.inputConnectors[0].connect(resize)
live.inputConnectors[0].connect(gray)
logi_gray = ensure(monochromeTOP,'ff_logi_gray')
logi_gray.inputConnectors[0].connect(root.op('camLogi'))
root.op('ff_live').inputConnectors[0].connect(logi_gray)
try:
    stop = getattr(root.op('ff_live_logic').module,'shutdown',None)
except Exception:
    stop = None
if stop:
    stop()
root.op('ff_live_logic').text = source
switch = root.op('final_switch')
if live not in switch.inputs:
    switch.inputConnectors[len(switch.inputs)].connect(live)
tick = ensure(executeDAT,'ff_live_tick')
tick.text = Path('C:/dev/ProjectSynchronicity/tools/ff/td_ff_tick.py').read_text()
tick.par.framestart = True
live.cook(force=True)
# Positions only for nodes created in this run — manual arrangements are preserved.
for node,x,y in ((resize,-360,-700),(gray,-180,-700),(logi_gray,-180,-500),
                 (logic,0,-700),(callbacks,180,-700),(live,360,-700),(tick,540,-700)):
    if node.name in created:
        node.nodeX,node.nodeY = x,y
root.op('win').par.winopen.pulse()
