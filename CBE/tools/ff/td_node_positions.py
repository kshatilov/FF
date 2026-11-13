"""Snapshot/restore /project1 node positions.
Run inside TD:  node tools/td/td.mjs --file tools/ff/td_node_positions.py
Restore:        node tools/td/td.mjs --exec "NODE_POSITIONS_MODE='restore';exec(open('tools/ff/td_node_positions.py').read())"
"""
import json
from pathlib import Path

PATH = Path('C:/dev/ProjectSynchronicity/tools/pipe/ff/td_node_positions.json')


def snapshot():
    root = op('/project1')
    data = {}
    for o in root.children:
        data[o.name] = dict(x=o.nodeX, y=o.nodeY, w=o.nodeWidth, h=o.nodeHeight)
    PATH.write_text(json.dumps(data, indent=1))
    return data


def restore():
    root = op('/project1')
    data = json.loads(PATH.read_text())
    moved = []
    for name, p in data.items():
        o = root.op(name)
        if o is None:
            continue
        if (o.nodeX, o.nodeY) != (p['x'], p['y']):
            moved.append(name)
        o.nodeX, o.nodeY = p['x'], p['y']
        if p.get('w') and p.get('h'):
            o.nodeWidth, o.nodeHeight = p['w'], p['h']
    return moved


if globals().get('NODE_POSITIONS_MODE') == 'restore':
    op('/project1').store('node_positions_restored', restore())
else:
    op('/project1').store('node_positions_snapshot', len(snapshot()))
