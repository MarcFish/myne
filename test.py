from graph import StaticGraph, TemporalGraph

math_path = "E:\project\data\process\mathoverflow\mathoverflow_5.csv"
dblp_path = "E:\project\data\process\dblp\dblp.csv"

sg = StaticGraph()
sg.read_from_file(dblp_path)
tg = TemporalGraph()
tg.read_from_file(math_path)
print(tg.get_node_neighbors(0,with_time=True))