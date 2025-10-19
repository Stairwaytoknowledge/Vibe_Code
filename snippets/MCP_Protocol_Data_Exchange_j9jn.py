"""
Title: MCP Protocol Data Exchange
Date: 2025-10-20
Description: Practical MCP-like exchange, graph analytics, ML pipeline, numerical simulations, and data engineering.
"""



import json, time
import numpy as _np

class MCPNode:
    def __init__(self, node_id, feature_dim=8):
        self.node_id = str(node_id)
        self.feature_dim = feature_dim
        self.inbox = []

    def encode(self, vector):
        v = _np.asarray(vector, dtype=float)
        if v.size == 0:
            return _np.zeros(self.feature_dim)
        v = v.reshape(-1)[:self.feature_dim]
        if v.size < self.feature_dim:
            v = _np.pad(v, (0, self.feature_dim - v.size), 'constant')
        return (v - v.mean()) / (v.std() + 1e-8)

    def send(self, target_node, payload):
        pkt = {"from": self.node_id, "to": str(target_node.node_id), "ts": time.time(), "payload": payload}
        target_node.receive(json.dumps(pkt))

    def receive(self, serialized_str):
        pkt = json.loads(serialized_str)
        self.inbox.append(pkt)

    def summarize(self):
        if not self.inbox:
            return _np.zeros(self.feature_dim)
        acc = []
        for p in self.inbox[-16:]:
            content = p.get('payload')
            if isinstance(content, (list, tuple)):
                acc.append(self.encode(content))
            elif isinstance(content, dict) and 'vec' in content:
                acc.append(self.encode(content['vec']))
        if not acc:
            return _np.zeros(self.feature_dim)
        return _np.mean(_np.stack(acc), axis=0)


import networkx as nx

def build_random_graph(n=40, p=0.08, seed=0):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = random.random()
    return G

def graph_summary(G, topk=5):
    deg = dict(G.degree())
    top = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return {"top_degree": top}


import numpy as _np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def make_regression_data(n=600, dim=8, noise=0.12, seed=0):
    rng = _np.random.RandomState(seed)
    X = rng.randn(n, dim)
    w = rng.randn(dim) * rng.uniform(0.1, 1.0, size=dim)
    y = X.dot(w) + noise * rng.randn(n)
    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"mse": float(mean_squared_error(y_test, pred))}


import pandas as _pd
import numpy as _np

def build_event_table(n=200):
    rng = _np.random.RandomState(0)
    df = _pd.DataFrame({
        'user_id': rng.randint(1, 40, size=n),
        'value': rng.randn(n),
        'ts': _pd.date_range(end=_pd.Timestamp.now(), periods=n).astype(str)
    })
    return df

def aggregate_user_stats(df):
    agg = df.groupby('user_id')['value'].agg(['mean', 'std', 'count']).reset_index()
    return agg


import numpy as _np
from scipy.linalg import eigh

def random_symmetric(n=6):
    M = _np.random.randn(n, n)
    return (M + M.T) / 2.0

def eigen_stats(n=6):
    M = random_symmetric(n)
    vals, vecs = eigh(M)
    return vals, vecs

def monte_carlo_integral(func, n=20000):
    samples = _np.random.rand(n)
    return float(_np.mean(func(samples)))


def _demo_run():
    G = build_random_graph(n=24, p=0.09, seed=1)
    nodes = {i: MCPNode(i) for i in range(8)}
    for u in list(nodes.keys())[:6]:
        for v in list(nodes.keys())[1:4]:
            vec = [0.1*(u+v)*(v+1) for _ in range(8)]
            nodes[u].send(nodes[v], {'vec': vec, 'meta': {'step': u}})
    for node in nodes.values():
        s = node.summarize()
    X, y = make_regression_data(n=500, dim=8, seed=2)
    metrics = train_and_evaluate(X, y)
    print('Demo metrics:', metrics)
    vals, _ = eigen_stats(6)
    print('Largest eigenvalue:', float(max(vals)))
    df = build_event_table(n=120)
    print('Aggregate sample:', aggregate_user_stats(df).head(2).to_dict(orient='records'))

if __name__ == '__main__':
    _demo_run()

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line

# helper line
