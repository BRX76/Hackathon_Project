import pandas as pd
import networkx as nx
test = pd.read_csv(r'C:\Users\dekel\Desktop\dmbi_results\train_1.csv')
test = test.where(test['is_click'] == 1)
test = test.loc[test['is_click'] == 1]
test = test[['Id', 'source_id', 'target_id']]
# test['size'] = test.groupby(['source_id', 'target_id']).size()
# test = test.groupby(['source_id', 'target_id']).size()
test = test.groupby(['source_id', 'target_id']).target_id.agg('count').to_frame('count').reset_index()
print(test)
# test_graph = nx.from_pandas_edgelist(test, 'source_id', 'target_id', create_using=nx.DiGraph())
# print('making graph now')
# print(nx.betweenness_centrality_source(test_graph))
G = nx.Graph()
G.add_nodes_from(test['source_id'] + test['target_id'])
G.add_edges_from(test['source_id', 'target_id'])
bw_centrality = nx.betweenness_centrality(G, normalized=False)
print(bw_centrality)


