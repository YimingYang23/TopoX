_base_ = ['./topox_stage2.py']

# Output adjacency refinement with the released TopoX weights.
data = dict(val=dict(output_geodist=True), test=dict(output_geodist=True))
