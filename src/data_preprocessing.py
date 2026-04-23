import pandas as pd
import os

# =========================
# LOAD DATA
# =========================
def load_data():
    BASE_DIR = os.path.dirname(__file__)
    RAW_PATH = os.path.join(BASE_DIR, "../data/raw/marvel_movies.csv")

    df = pd.read_csv(RAW_PATH, encoding='latin1')
    print("Initial shape:", df.shape)

    return df


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df[['Title', 'Cast', 'Genres', 'Plot', 'Release Date', 'Status']]

    df.columns = ['title', 'cast', 'genres', 'plot', 'release_date', 'status']

    df = df.drop_duplicates(subset='title')
    df = df.dropna(subset=['title'])

    df['title'] = df['title'].str.strip().str.title()
    df['plot'] = df['plot'].fillna("").str.lower()

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # ✅ CLEAN LISTS PROPERLY
    def clean_list(x):
        if pd.isna(x) or x == "":
            return []

        x = str(x)
        x = x.replace("[", "").replace("]", "").replace("'", "")

        return [i.strip() for i in x.split(",") if i.strip()]

    df['cast'] = df['cast'].apply(clean_list)
    df['genres'] = df['genres'].apply(clean_list)

    # Fix encoding
    df['cast'] = df['cast'].apply(lambda x: [i.encode('ascii', 'ignore').decode() for i in x])

    df['status'] = df['status'].str.lower()

    return df


# =========================
# FIX INCORRECT MOVIES
# =========================
def fix_incorrect_entries(df):
    corrections = {
        "Untitled Spider-Man: Far From Home Sequel": "Spider-Man: No Way Home",
        "Black Panther Ii": "Black Panther: Wakanda Forever",
        "Captain Marvel 2": "The Marvels"
    }

    df['title'] = df['title'].replace(corrections)

    return df


# =========================
# UPDATE STATUS
# =========================
def update_status(df):
    today = pd.Timestamp.today()

    df['status'] = df['release_date'].apply(
        lambda x: 'released' if pd.notnull(x) and x <= today else 'upcoming'
    )

    return df


# =========================
# ADD MISSING MOVIES
# =========================
def add_missing_movies(df):
    new_movies = [
        {"title": "Deadpool & Wolverine", "release_date": "2024-07-26", "status": "released","cast": [],"genres": [],"plot": "unknown"},
        {"title": "Captain America: Brave New World", "release_date": "2025-02-14", "status": "upcoming","cast": [],"genres": [],"plot": "unknown"},
        {"title": "Thunderbolts", "release_date": "2025-07-25", "status": "upcoming","cast": [],"genres": [],"plot": "unknown"},
        {"title": "Fantastic Four", "release_date": "2025-11-07", "status": "upcoming","cast": [],"genres": [],"plot": "unknown"},
        {"title": "Spider-Man: Brand New Day", "release_date": "2026-07-31", "status": "upcoming","cast": [],"genres": [],"plot": "unknown"},
        {"title": "Avengers: Doomsday", "release_date": "2026-12-18", "status": "upcoming","cast": [],"genres": [],"plot": "unknown"},
        {"title": "Avengers: Secret Wars", "release_date": "2027-05-07", "status": "upcoming","cast": [],"genres": [],"plot": "unknown"}
    ]

    new_df = pd.DataFrame(new_movies)
    new_df['release_date'] = pd.to_datetime(new_df['release_date'])

    df = pd.concat([df, new_df], ignore_index=True)

    return df


# =========================
# FINAL CLEANUP
# =========================
def finalize_data(df):
    df = df.drop_duplicates(subset='title')
    df = df.sort_values(by='release_date')
    df = df.reset_index(drop=True)

    return df


# =========================
# SAVE DATA
# =========================
def save_data(df):
    BASE_DIR = os.path.dirname(__file__)
    SAVE_PATH = os.path.join(BASE_DIR, "../data/processed/final_marvel_dataset.csv")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    df.to_csv(SAVE_PATH, index=False)
    print("Final dataset saved at:", SAVE_PATH)


# =========================
# MAIN
# =========================
def main():
    df = load_data()

    df = clean_data(df)
    df = fix_incorrect_entries(df)
    df = update_status(df)
    df = add_missing_movies(df)
    df['plot'] = df['plot'].fillna("unknown")

    df = finalize_data(df)

    save_data(df)

    print("\nFinal Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()