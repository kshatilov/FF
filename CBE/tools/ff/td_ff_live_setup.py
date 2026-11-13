from pathlib import Path

root = op('/project1')
created = set()
def ensure(node_type, name):
    node = root.op(name)
    if node is None:
        node = root.create(node_type, name)
        created.add(name)
    return node

logic = ensure(textDAT,'ff_live_logic')
try:
    stop = getattr(logic.module,'shutdown',None)
except Exception:
    stop = None
if stop:
    stop()
logic.text = Path('C:/dev/ProjectSynchronicity/tools/ff/td_ff_live.py').read_text()
callbacks = ensure(textDAT,'ff_live_callbacks')
callbacks.text = "def onCook(scriptOp):\n    op('/project1/ff_live_logic').module.cook(scriptOp)\n"
live = ensure(scriptTOP,'ff_live')
live.par.callbacks = callbacks.path
gray = ensure(monochromeTOP,'ff_logi_gray')
gray.inputConnectors[0].connect(root.op('camLogi'))
live.inputConnectors[0].connect(gray)
switch = root.op('final_switch')
if live not in switch.inputs:
    switch.inputConnectors[len(switch.inputs)].connect(live)
tick = ensure(executeDAT,'ff_live_tick')
tick.text = Path('C:/dev/ProjectSynchronicity/tools/ff/td_ff_tick.py').read_text()
tick.par.framestart = True
# Positions only for nodes created in this run — manual arrangements are preserved.
for node,x,y in ((gray,-180,-500),(logic,0,-500),(callbacks,180,-500),(live,360,-500),(tick,540,-700)):
    if node.name in created:
        node.nodeX,node.nodeY = x,y
live.cook(force=True)
