from torchview import draw_graph
from uresnet_sparse import * 
from graphviz import Digraph

model = Net()

for name, module in model.named_children():
    print(f"{name}: {module.__class__.__name__}")


graph = draw_graph(
    model,
    input_size=(1, 1, 28, 28),
    expand_nested=True,   # show submodules (ResidualSparseBlock2d, etc.)
    depth=1,              # how deep to go
)

graph.visual_graph.render("uresnet_graph", format="png")

# --------------------------

dot = Digraph(comment='U-ResNet Sparse Model')

dot.node('E1', 'Encoder 1\nResidualSparseBlock2d')
dot.node('E2', 'Encoder 2\nResidualSparseBlock2d')
dot.node('B', 'Bottleneck\nAttention')
dot.node('U1', 'Decoder Up1\nConvTrBlock2d')
dot.node('D1', 'Decoder 1\nResidualSparseBlock2d')
dot.node('U0', 'Decoder Up0\nConvTrBlock2d')
dot.node('D0', 'Decoder 0\nResidualSparseBlock2d')
dot.node('H', 'Head\nDense classifier')

# Flow edges
dot.edges([
    ('E1','E2'), 
    ('E2','B'), 
    ('B','U1'), 
    ('U1','D1'), 
    ('D1','U0'), 
    ('U0','D0'), 
    ('D0','H')
])

# Skip connections
dot.edge('E1', 'D0', style='dashed')
dot.edge('E2', 'D1', style='dashed')

dot.render('uresnet_skeleton', format='png', cleanup=True)


