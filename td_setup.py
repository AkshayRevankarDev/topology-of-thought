import sys
sys.path.insert(0, '/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought')

p = root

# Clear canvas
for o in p.findChildren(maxDepth=1):
    if o.name not in ('perform', 'local'):
        try: o.destroy()
        except: pass

# Video Device In TOP
vid = p.create(videodeviceInTOP, 'videodevin1')
vid.par.flipx = True
vid.nodeX, vid.nodeY = -400, 200

# Table DATs
nd = p.create(tableDAT, 'nodes')
nd.nodeX, nd.nodeY = -400, 0
nd.clear()
nd.appendRow(['id','label','x','y','tx','ty','vx','vy','energy','confidence','hub','paper_source'])

ed = p.create(tableDAT, 'edges')
ed.nodeX, ed.nodeY = -400, -120
ed.clear()
ed.appendRow(['id','node_a','node_b','label','confidence','paper_source','is_bridge'])

# Text DATs
for name, content in [('current_mode','0'),('pdf_path',''),('query_answer',''),('gesture_state','IDLE')]:
    d = p.create(textDAT, name)
    d.text = content

# Script CHOP
hc = p.create(scriptCHOP, 'hand_tracking')
hc.nodeX, hc.nodeY = 0, 200
hc.text = open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/hand_tracking.py').read()

# Script TOP
gt = p.create(scriptTOP, 'graph_renderer')
gt.nodeX, gt.nodeY = 0, 0
gt.par.resolutionw = 1280
gt.par.resolutionh = 720
gt.par.format = 'rgba8fixed'
gt.text = open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/graph_renderer.py').read()

# Execute DAT
ex = p.create(executeDAT, 'main_execute')
ex.nodeX, ex.nodeY = 200, 0
ex.par.framestart = True
ex.par.active = True
ex.text = open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/physics.py').read()

# Load session data
from core.graph_state import GraphState
import json
with open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/data/sessions/attention_is_all_you_need.json') as f:
    graph = GraphState.from_json(f.read())
nd.clear(keepFirstRow=True)
for n in graph.nodes:
    nd.appendRow([n.id, n.label, 640, 360, 640, 360, 0, 0, n.energy, n.confidence, 1 if n.hub else 0, n.paper_source])
ed.clear(keepFirstRow=True)
for e in graph.edges:
    ed.appendRow([e.id, e.node_a, e.node_b, e.label, e.confidence, e.paper_source, 1 if e.is_bridge else 0])

# Over TOP
ov = p.create(overTOP, 'compositor')
ov.nodeX, ov.nodeY = 400, 100
ov.inputConnectors[0].connect(vid)
ov.inputConnectors[1].connect(gt)

# Out TOP
out = p.create(outTOP, 'output')
out.nodeX, out.nodeY = 600, 100
out.inputConnectors[0].connect(ov)

print('=== DONE ===')
print('nodes:', nd.numRows, 'edges:', ed.numRows)
