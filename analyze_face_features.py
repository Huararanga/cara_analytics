import pandas as pd

# Load the face_features data
face_features = pd.read_parquet("./face_features.parquet")

# Count photos per person (id_person)
photos_per_person = face_features['id_person'].value_counts()

print("Number of photos per person:")
print(photos_per_person)

print(f"\nTotal number of people: {len(photos_per_person)}")
print(f"Total number of photos: {len(face_features)}")
print(f"Average photos per person: {len(face_features) / len(photos_per_person):.2f}")

print(f"\nMinimum photos per person: {photos_per_person.min()}")
print(f"Maximum photos per person: {photos_per_person.max()}")

# Show distribution
print(f"\nDistribution of photos per person:")
print(photos_per_person.value_counts().sort_index()) 