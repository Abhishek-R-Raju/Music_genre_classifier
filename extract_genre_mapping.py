import os

def extract_genre_mapping(data_path="Data/genres_original"):
    """Extract genre mapping from the dataset directory and save it to a file."""
    genre_mapping = []
    
    # Walk through the directories to find genres
    for _, dirnames, _ in os.walk(data_path):
        # Skip the root directory
        break
    
    # Sort the directories to ensure consistent mapping
    dirnames.sort()
    
    for dirname in dirnames:
        genre_mapping.append(dirname)
    
    # Save the mapping to a file
    with open('genre_mapping.txt', 'w') as f:
        for genre in genre_mapping:
            f.write(f"{genre}\n")
    
    print(f"Saved {len(genre_mapping)} genres to genre_mapping.txt: {genre_mapping}")
    return genre_mapping

if __name__ == "__main__":
    extract_genre_mapping()