import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from graph_builder import build_graph


# =========================
# LOAD DATA
# =========================
def load_data(path):
    df = pd.read_csv(path)
    return df


# =========================
# BUILD CONTENT MODEL (NLP)
# =========================
def build_content_model(df):
    tfidf = TfidfVectorizer(stop_words='english')

    df['plot'] = df['plot'].fillna("")
    tfidf_matrix = tfidf.fit_transform(df['plot'])

    similarity = cosine_similarity(tfidf_matrix)

    return similarity


# =========================
# CONTENT-BASED RECOMMENDER
# =========================
def recommend_by_plot(title, df, similarity, top_n=5):
    if title not in df['title'].values:
        return []

    idx = df[df['title'] == title].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [df.iloc[i[0]]['title'] for i in scores[1:top_n+1]]


# =========================
# GRAPH-BASED RECOMMENDER
# =========================
def recommend_by_graph(G, character, top_n=5):
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
# HYBRID RECOMMENDER 🔥
# =========================
def hybrid_recommend(title, df, similarity, alpha=0.6, top_n=5):
    if title not in df['title'].values:
        return []

    idx = df[df['title'] == title].index[0]

    content_scores = list(enumerate(similarity[idx]))
    max_score = max([score for _, score in content_scores])
    content_scores = [(i, score / max_score) for i, score in content_scores]

    results = []

    base_cast = df.iloc[idx]['cast']
    if isinstance(base_cast, str):
        base_cast = base_cast.strip("[]").replace("'", "").split(", ")

    for i, score in content_scores:
        movie = df.iloc[i]['title']

        movie_cast = df.iloc[i]['cast']
        if isinstance(movie_cast, str):
            movie_cast = movie_cast.strip("[]").replace("'", "").split(", ")

        overlap = set(base_cast) & set(movie_cast)
        graph_score = len(overlap) / max(1, len(base_cast))

        final_score = (alpha * score) + ((1 - alpha) * graph_score)

        # 🧠 AI EXPLANATION
        explanation = ""
        if len(overlap) > 0:
            explanation = f"Shares cast with {title}: {', '.join(list(overlap)[:2])}"
        else:
            explanation = "Similar storyline and genre"

        results.append({
            "title": movie,
            "score": round(final_score * 100),
            "explanation": explanation
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[1:top_n+1]

    # Sort results
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return [r[0] for r in results[1:top_n+1]]
def recommend_from_character(character, G, mapping, top_n=5):
    if character not in mapping:
        return []

    actor = mapping[character]

    if actor not in G:
        return []

    neighbors = G[actor]

    sorted_neighbors = sorted(
        neighbors.items(),
        key=lambda x: x[1]['weight'],
        reverse=True
    )

    return [char for char, _ in sorted_neighbors[:top_n]]

character_to_actor = {
    "Iron Man": "Robert Downey Jr.",
    "Captain America": "Chris Evans",
    "Thor": "Chris Hemsworth",
    "Black Widow": "Scarlett Johansson",
    "Hulk": "Mark Ruffalo"
}

# =========================
# MAIN
# =========================
def main():
    df = load_data("data/processed/final_marvel_dataset.csv")

    # ✅ ADD HERE
    df = df[df['status'] == 'released']

    similarity = build_content_model(df)

    G = build_graph(df)

    print("\nPlot-based recommendations:")
    print(recommend_by_plot("Iron Man", df, similarity))

    print("\nGraph-based recommendations:")
    print(recommend_by_graph(G, "Robert Downey Jr."))

    print("\nHybrid recommendations:")
    print(hybrid_recommend("Iron Man", df, similarity))

    print("\nCharacter-based recommendations:")
    print(recommend_from_character("Iron Man", G, character_to_actor))


if __name__ == "__main__":
    main()