root = op('/project1')
# Positions are applied once only: the user's manual arrangement is preserved
# on every subsequent run. Delete the stored key to re-apply the layout.
placed = root.fetch('ff_layout_positions_done', False)
layout = (
    ('anim_controls',1725,-400,(0.23,0.43,0.68)),
    ('anim_phase',1960,-400,(0.72,0.48,0.20)),
    ('anim_source',1725,-650,(0.45,0.32,0.65)),
    ('anim_height',1960,-900,(0.20,0.52,0.54)),
    ('anim_markers',1960,-1080,(0.20,0.52,0.54)),
    ('anim_layers',2185,-650,(0.27,0.58,0.35)),
    ('animation_pipe',2410,-650,(0.27,0.58,0.35)),
)
for name,x,y,color in layout:
    node = root.op(name)
    if node is None:
        continue
    if not placed:
        node.nodeX,node.nodeY = x,y
    node.color = color
    node.viewer = False
if not placed:
    for child,parent in (('anim_buttons','anim_controls'),('anim_height_code','anim_height'),
                         ('anim_markers_code','anim_markers'),('anim_layers_code','anim_layers')):
        node,owner = root.op(child),root.op(parent)
        if node is not None and owner is not None:
            node.dock = owner
            node.nodeX,node.nodeY = 0,-120
for name in ('anim_controls','anim_height','anim_layers'):
    node = root.op(name)
    if node is not None:
        node.showDocked = False
box = root.op('anim_group')
if box is None:
    box = next(iter(root.findChildren(depth=1,tags=['ff_animation_layout'],includeUtility=True)),None)
if box is None:
    box = root.create(annotateCOMP,'anim_group')
box.tags.add('ff_animation_layout')
if not placed:
    box.par.encloseops = False
box.par.Mode = 'networkbox'
box.name = 'anim_group'
box.par.Titletext = 'PYRAMID BUILD  |  Source + Height + Markers > Layers > Output'
box.par.Bodytext = ''
box.par.Titleheight = 48
box.par.Backcolorr,box.par.Backcolorg,box.par.Backcolorb = 0.13,0.19,0.26
box.par.Backcoloralpha = 0.65
box.par.utility = True
if not placed:
    box.nodeX,box.nodeY = 1650,-1100
    box.nodeWidth,box.nodeHeight = 950,900
    box.par.encloseops = True
    root.store('ff_layout_positions_done', True)
