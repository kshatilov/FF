from pathlib import Path
import json

HERE = Path('C:/dev/ProjectSynchronicity/tools/ff')
PIPE_FF = Path('C:/dev/ProjectSynchronicity/tools/pipe/ff')
root = op('/project1')
controls = root.op('anim_controls') or root.create(baseCOMP,'anim_controls')
page = controls.appendCustomPage('Build')
if not hasattr(controls.par,'Tracking'):
    p = page.appendMenu('Tracking',label='Tracking Camera')[0]
    p.menuNames = ['cam','kinect']
    p.menuLabels = ['Logi Camera','Kinect Color']
    p.default = 'kinect'
    p.val = 'kinect'
for name,label,value,kind,minimum,maximum in (
        ('Layers','Contour Bands',6,'int',1,64),
        ('Buildseconds','Breath Seconds',5.,'float',0.5,60),
        ('Holdseconds','Color Cycle Seconds',24.,'float',2,120),
        ('Groove','Wave Amount',0.45,'float',0,1),
        ('Brightness','Brightness',1.2,'float',0,3)):
    if not hasattr(controls.par,name):
        p = (page.appendInt(name,label=label) if kind=='int' else page.appendFloat(name,label=label))[0]
        p.default,p.val = value,value
    p = getattr(controls.par,name)
    p.label,p.default = label,value
    p.min,p.max,p.normMin,p.normMax = minimum,maximum,minimum,maximum
    p.clampMin,p.clampMax = True,True
for name,value in (('Layers',6),('Buildseconds',5.),('Holdseconds',24.),('Groove',0.45)):
    getattr(controls.par,name).val = value
if not hasattr(controls.par,'Play'):
    p = page.appendToggle('Play',label='Play Animation')[0]
    p.default,p.val = True,True
if not hasattr(controls.par,'Restart'):
    page.appendPulse('Restart',label='Restart Breath')
controls.par.Restart.label = 'Restart Breath'
buttons = root.op('anim_buttons')
if buttons is None:
    buttons = root.create(parameterexecuteDAT,'anim_buttons')
    buttons.nodeX,buttons.nodeY = 900,-420
buttons.text = "def onPulse(par):\n    op('/project1/anim_phase').par.resetpulse.pulse()\n    op('/project1/anim_epoch').par.const0value = absTime.seconds\n"
buttons.par.op = controls.path
buttons.par.pars = 'Restart'
buttons.par.valuechange = False
buttons.par.onpulse = True
source = root.op('anim_source') or root.create(switchTOP,'anim_source')
source.inputConnectors[0].connect(root.op('ff_live_cam1') or root.op('ff_live'))
source.inputConnectors[1].connect(root.op('ff_live_kinect'))
source.par.index.expr = "op('anim_controls').par.Tracking.menuIndex"
source.par.blend = False
phase = root.op('anim_phase') or root.create(lfoCHOP,'anim_phase')
phase.par.wavetype = 'sin'
phase.par.channelname = 'build'
phase.par.amp,phase.par.offset = 0.5,0.5
phase.par.frequency.expr = "1/op('anim_controls').par.Buildseconds"
phase.par.play.expr = "op('anim_controls').par.Play"
epoch = root.op('anim_epoch') or root.create(constantCHOP,'anim_epoch')
epoch.par.const0name = 't0'
height = root.op('anim_height') or root.create(glslTOP,'anim_height')
layers = root.op('anim_layers') or root.create(glslTOP,'anim_layers')
markers = root.op('anim_markers')
if markers is None:
    markers = root.create(glslTOP,'anim_markers')
    markers.nodeX,markers.nodeY = 1960,-1080
for node,filename in ((height,'ff_build_height.frag'),(layers,'ff_build_layers.frag'),
                      (markers,'ff_build_markers.frag')):
    code = root.op(node.name+'_code')
    if code is None:
        code = root.create(textDAT,node.name+'_code')
        code.nodeX,code.nodeY = 900,-900 if node==height else (-1080 if node==markers else -1060)
    code.text = (HERE/filename).read_text()
    node.par.pixeldat = code.path
    node.par.outputresolution = 'custom'
    node.par.resolutionw,node.par.resolutionh = 1280,720
measured = json.loads((PIPE_FF/'fulcrum_spec.json').read_text())['measured']
for shader in (height,markers):
    shader.par.format = 'rgba16float'
    for i,name in enumerate(('uOrigin','uRayX','uRayY','uRayZ','uGeometry')):
        getattr(shader.par,f'vec{i}name').val = name
    shader.par.vec0valuew = 0
    shader.par.vec4valuex = measured['height']
    shader.par.vec4valuey = measured['base_edge']/2
    shader.par.vec4valuez = measured['top_edge']/2
layers.inputConnectors[0].connect(source)
layers.inputConnectors[1].connect(height)
layers.inputConnectors[2].connect(markers)
layers.par.vec0name = 'uBuild'
layers.par.vec0valuex.expr = "op('anim_phase')['build']"
layers.par.vec0valuey.expr = "op('anim_controls').par.Layers"
layers.par.vec0valuez.expr = "op('anim_controls').par.Groove"
layers.par.vec0valuew.expr = "op('anim_controls').par.Brightness"
layers.par.vec1name = 'uTiming'
layers.par.vec1valuex.expr = "op('anim_controls').par.Holdseconds"
layers.par.vec1valuey.expr = "absTime.seconds - op('anim_epoch')['t0']"
table = root.op('anim_table')
if table is None:
    table = root.create(glslTOP,'anim_table')
    table.nodeX,table.nodeY = 2140,-1180
table_code = root.op('anim_table_code')
if table_code is None:
    table_code = root.create(textDAT,'anim_table_code')
    table_code.nodeX,table_code.nodeY = 2140,-1280
table_code.text = (HERE/'ff_anim_table.frag').read_text()
table.par.pixeldat = table_code.path
table.par.outputresolution = 'custom'
table.par.resolutionw,table.par.resolutionh = 1280,720
table.par.format = 'rgba16float'
for i,name in enumerate(('uOrigin','uRayX','uRayY','uRayZ','uTW0','uTW1','uTW2')):
    getattr(table.par,f'vec{i}name').val = name
table.par.vec0valuew = 0
tetris = root.op('anim_tetris')
if tetris is None:
    tetris = root.create(glslTOP,'anim_tetris')
    tetris.nodeX,tetris.nodeY = 2320,-1060
tetris_code = root.op('anim_tetris_code')
if tetris_code is None:
    tetris_code = root.create(textDAT,'anim_tetris_code')
    tetris_code.nodeX,tetris_code.nodeY = 2320,-1160
tetris_code.text = (HERE/'ff_build_tetris.frag').read_text()
tetris.par.pixeldat = tetris_code.path
tetris.par.outputresolution = 'custom'
tetris.par.resolutionw,tetris.par.resolutionh = 1280,720
tetris.inputConnectors[0].connect(height)
tetris.inputConnectors[1].connect(markers)
tetris.inputConnectors[2].connect(table)
tetris.par.vec0name = 'uBuild'
tetris.par.vec0valuex.expr = "op('anim_phase')['build']"
tetris.par.vec0valuew.expr = "op('anim_controls').par.Brightness"
tetris.par.vec1name = 'uTiming'
tetris.par.vec1valuex.expr = "op('anim_controls').par.Buildseconds"
tetris.par.vec1valuey.expr = "absTime.seconds - op('anim_epoch')['t0']"
for i,name in enumerate(('uTW0','uTW1','uTW2')):
    getattr(tetris.par,f'vec{i+2}name').val = name
pipe = root.op('animation_pipe') or root.create(nullTOP,'animation_pipe')
pipe.inputConnectors[0].connect(layers)
tetris_out = root.op('animation_tetris') or root.create(nullTOP,'animation_tetris')
tetris_out.inputConnectors[0].connect(tetris)
switch = root.op('final_switch')
if pipe not in switch.inputs:
    switch.inputConnectors[len(switch.inputs)].connect(pipe)
if tetris_out not in switch.inputs:
    switch.inputConnectors[len(switch.inputs)].connect(tetris_out)
tick = root.op('ff_live_tick')
tick.text = (HERE/'td_ff_tick.py').read_text()
source_code = (HERE/'td_ff_live.py').read_text()
for name,kinect in (('ff_live_logic',False),('ff_live_kinect_logic',True)):
    node = root.op(name)
    node.module.shutdown()
    node.text = source_code+ ('\nconfigure_kinect()\n' if kinect else '')
exec((HERE/'td_ff_animation_layout.py').read_text())
phase.par.resetpulse.pulse()
root.op('win').par.winopen.pulse()
