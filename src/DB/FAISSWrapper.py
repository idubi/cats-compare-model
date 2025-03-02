import faiss
import numpy as np
import os
import json

#FAISS (Facebook AI Similarity Search) is a library
#----------------------------------------------------- 
#for efficient similarity search
#and clustering of dense vectors. 
#It contains algorithms that search in sets of vectors of any size,
#up to ones that possibly do not fit in RAM.
#It also contains supporting code for evaluation and parameter tuning.

#https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c


class FAISSWrapper:
    def __init__(self, dimension=128, index_file="faiss.index", metadata_file="metadata.json"):
        """
        Initialize FAISS with file-based persistence.
        :param dimension: Length of feature vectors (default 128).
        :param index_file: File to save/load FAISS index.
        :param metadata_file: File to save/load metadata.
        """
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.pet_db = {}  # Store pet metadata (pet name, species)
        
        # Initialize FAISS index
        if os.path.exists(index_file):
            print("🔄 Loading FAISS index from file...")
            self.index = faiss.read_index(index_file)
            self.load_metadata()
        else:
            print("🆕 No FAISS index found. Creating a new one.")
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
            self.save_index()

    def add_pet(self, pet_id, name, species, embedding):
        """
        Add a pet embedding to FAISS and save metadata.
        :param pet_id: Unique ID for the pet
        :param name: Pet name
        :param species: Pet species (e.g., "dog", "cat")
        :param embedding: The feature vector (numpy array)
        """
        embedding = np.expand_dims(embedding, axis=0).astype("float32")  # Ensure correct format
        self.index.add(embedding)  # Add to FAISS
        self.pet_db[pet_id] = {"name": name, "species": species}  # Store metadata
        print(f"✅ Added {name} ({species}) to FAISS.")
        
        # Save FAISS and metadata
        self.save_index()
        self.save_metadata()

    def find_closest_match(self, embedding, top_k=1):
        """
        Find the closest matching pet in FAISS.
        :param embedding: The input feature vector (numpy array)
        :param top_k: Number of closest matches to retrieve
        :return: List of top matches with distance scores
        """
        embedding = np.expand_dims(embedding, axis=0).astype("float32")
        D, I = self.index.search(embedding, top_k)  # Search FAISS
        matches = []

        for i in range(top_k):
            pet_id = str(I[0][i])
            if pet_id in self.pet_db:
                pet_info = self.pet_db[pet_id]
                matches.append({
                    "name": pet_info["name"],
                    "species": pet_info["species"],
                    "distance": D[0][i]
                })

        return matches

    def save_index(self):
        """ Saves FAISS index to file. """
        faiss.write_index(self.index, self.index_file)
        print("💾 FAISS index saved.")

    def save_metadata(self):
        """ Saves metadata to a JSON file. """
        with open(self.metadata_file, "w") as f:
            json.dump(self.pet_db, f)
        print("💾 Metadata saved.")

    def load_metadata(self):
        """ Loads metadata from a JSON file. """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.pet_db = json.load(f)
            print("📂 Metadata loaded.")
        else:
            print("❌ No metadata file found. Starting fresh.")

    def get_total_pets(self):
        """ Returns the total number of stored embeddings. """
        return self.index.ntotal

# Example Usage
if __name__ == "__main__":
    faiss_wrapper = FAISSWrapper(dimension=128)

    # Add some pets
    faiss_wrapper.add_pet(0, "Fluffy", "cat", np.random.rand(128).astype("float32"))
    faiss_wrapper.add_pet(1, "Rex", "dog", np.random.rand(128).astype("float32"))

    # Search for a similar pet
    new_embedding = np.random.rand(128).astype("float32")
    match = faiss_wrapper.find_closest_match(new_embedding)
    print("Closest Match:", match)

    # Get total pets stored
    print(f"Total pets stored: {faiss_wrapper.get_total_pets()}")
