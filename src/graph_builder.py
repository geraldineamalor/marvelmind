import pandas as pd
import networkx as nx
from itertools import combinations


# =========================
# LOAD CLEAN DATA
# =========================
def load_data(path):
    df = pd.read_csv(path)
    return df


# =========================
# BUILD GRAPH
# =========================
def build_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        cast = row['cast']

        # Convert string back to list (because CSV saves lists as strings)
        if isinstance(cast, str):
            cast = cast.strip("[]").replace("'", "").split(", ")

        # Create edges between all pairs
        for actor1, actor2 in combinations(cast, 2):
            if G.has_edge(actor1, actor2):
                G[actor1][actor2]['weight'] += 1
            else:
                G.add_edge(actor1, actor2, weight=1)

    return G


# =========================
# BASIC ANALYSIS
# =========================
def analyze_graph(G):
    print("Number of characters:", G.number_of_nodes())
    print("Number of connections:", G.number_of_edges())

    # Top connected characters
    degrees = dict(G.degree())
    top_chars = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\nTop connected characters:")
    for char, deg in top_chars:
        print(char, "->", deg)


# =========================
# GET CONNECTIONS
# =========================
def get_connections(G, character):
    if character in G:
        neighbors = G[character]
        return sorted(neighbors.items(), key=lambda x: x[1]['weight'], reverse=True)
    else:
        return []

def recommend_characters(G, character, top_n=5):
    if character not in G:
        return []

    neighbors = G[character]

    sorted_neighbors = sorted(
        neighbors.items(),
        key=lambda x: x[1]['weight'],
        reverse=True
    )

    return [char for char, _ in sorted_neighbors[:top_n]]

# =========================
# MAIN
# =========================
def main():
    df = load_data("data/processed/final_marvel_dataset.csv")

    G = build_graph(df)

    analyze_graph(G)

    # Example
    print("\nConnections of Robert Downey Jr.:")
    connections = get_connections(G, "Robert Downey Jr.")
    
    for char, data in connections[:5]:
        print(char, "-> weight:", data['weight'])

    print("\nRecommended for Robert Downey Jr.:")
    recs = recommend_characters(G, "Robert Downey Jr.")

    for r in recs:
        print(r)


if __name__ == "__main__":
    main()