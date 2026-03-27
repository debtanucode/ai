import os
from src.search import MovieSearchEngine

CSV_PATH = os.path.join(os.path.dirname(__file__), "indian_movies.csv")


def display_results(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    for movie in results:
        name = movie.get("Movie Name") or movie.get("Name", "Unknown")
        print(f"#{movie['rank']}  {name}  ({movie['Release Date']})")
        print(f"    IMDB: {movie['IMDB Rating']}  |  Distance: {movie['distance']}")
        print(f"    {movie['Description'][:120]}...")
        print()


def main():
    engine = MovieSearchEngine(csv_path=CSV_PATH)

    print("=" * 60)
    print("  Indian Movie Semantic Search (Dense Retrieval)")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nEnter your search query: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        results = engine.search(query, top_k=5)
        display_results(results)


if __name__ == "__main__":
    main()
