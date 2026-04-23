from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import ast
from recommender import build_content_model, hybrid_recommend, recommend_by_plot
from graph_builder import build_graph, recommend_characters

app = Flask(__name__)
CORS(app)

# =========================
# LOAD & PREPARE DATASET
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/final_marvel_dataset.csv")

df = pd.read_csv(DATA_PATH, encoding="latin1")

def fix_list_column(col):
    def parse(x):
        try:
            return ast.literal_eval(x)
        except:
            x = str(x).replace("[", "").replace("]", "").replace("'", "")
            return [i.strip() for i in x.split(",") if i.strip()]
    return col.apply(parse)

df['cast']   = fix_list_column(df['cast'])
df['genres'] = fix_list_column(df['genres'])
df['plot']   = df['plot'].fillna("")

# =========================
# BUILD MODELS ONCE ON START
# =========================
print("[MarvelMind] Building TF-IDF similarity matrix...")
released_df  = df[df['status'] == 'released'].reset_index(drop=True)
similarity   = build_content_model(released_df)

print("[MarvelMind] Building cast graph...")
G = build_graph(released_df)

print(f"[MarvelMind] Ready — {len(df)} films loaded, {len(released_df)} released.")

# =========================
# HELPER: serialise a row
# =========================
def serialise_film(row):
    return {
        "title":        row['title'],
        "year":         str(row['release_date'])[:4] if pd.notnull(row.get('release_date')) else "N/A",
        "genres":       row['genres'] if isinstance(row['genres'], list) else [],
        "cast":         row['cast']   if isinstance(row['cast'],   list) else [],
        "plot":         row['plot'],
        "status":       row.get('status', 'released'),
        "phase":        int(row['phase']) if pd.notnull(row.get('phase')) else None,
    }

# =========================
# GET ALL FILMS
# =========================
@app.route('/api/films', methods=['GET'])
def get_all_films():
    records = [serialise_film(row) for _, row in df.iterrows()]
    return jsonify(records)

# =========================
# MOVIE RECOMMENDATIONS
# =========================
@app.route('/recommend/movie', methods=['POST'])
def recommend_movie():
    data  = request.json or {}
    title = data.get('movie', '').strip()
    alpha = float(data.get('alpha', 0.6))   # plot weight 0–1
    top_n = int(data.get('top_n', 5))

    # Fuzzy match title (case-insensitive)
    match = released_df[released_df['title'].str.lower() == title.lower()]
    if match.empty:
        match = released_df[released_df['title'].str.lower().str.contains(title.lower(), na=False)]
    if match.empty:
        return jsonify({"error": f"Film '{title}' not found in dataset."}), 404

    exact_title = match.iloc[0]['title']
    results = hybrid_recommend(exact_title, released_df, similarity, alpha=alpha, top_n=top_n)

    # Enrich with full film data
    enriched = []
    for rec in results:
        row_match = df[df['title'] == rec['title']]
        if not row_match.empty:
            film_data = serialise_film(row_match.iloc[0])
            enriched.append({**film_data, **rec})
        else:
            enriched.append(rec)

    return jsonify(enriched)

# =========================
# CHARACTER / ACTOR SEARCH
# =========================
@app.route('/recommend/character', methods=['POST'])
def recommend_character():
    data  = request.json or {}
    query = data.get('character', '').strip().lower()
    top_n = int(data.get('top_n', 5))

    if not query:
        return jsonify({"error": "No character name provided."}), 400

    # Find all films featuring this actor/character
    appearances = []
    for _, row in df.iterrows():
        cast_list = row['cast'] if isinstance(row['cast'], list) else []
        matched_actors = [a for a in cast_list if query in a.lower()]
        if matched_actors:
            appearances.append({
                **serialise_film(row),
                "matched_actor": matched_actors[0]
            })

    if not appearances:
        return jsonify({"error": f"No films found for '{query}'."}), 404

    # Also return graph-based co-star recommendations
    actor_name = appearances[0]['matched_actor']
    costars = recommend_characters(G, actor_name, top_n=top_n)

    return jsonify({
        "actor":       actor_name,
        "films":       appearances,
        "costars":     costars,
        "film_count":  len(appearances),
        "phase_count": len(set(a['phase'] for a in appearances if a.get('phase')))
    })

# =========================
# FILM DETAIL
# =========================
@app.route('/api/films/<path:title>', methods=['GET'])
def get_film(title):
    match = df[df['title'].str.lower() == title.lower()]
    if match.empty:
        return jsonify({"error": "Film not found"}), 404
    return jsonify(serialise_film(match.iloc[0]))

# =========================
# HEALTH CHECK
# =========================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":          "ok",
        "total_films":     len(df),
        "released_films":  len(released_df),
        "graph_nodes":     G.number_of_nodes(),
        "graph_edges":     G.number_of_edges(),
    })

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
