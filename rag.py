import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


with open("./database/disease_database.json", "r", encoding="utf-8") as f:
    database = json.load(f)


class CropDiseaseRAG:
    def __init__(self):
        """Initialize RAG system using sklearn instead of FAISS for reliability."""
        self.database = database
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.documents = []
        self.doc_embeddings = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Create embeddings for all database entries."""
        print("Setting up embeddings...")
        
        # Prepare text documents
        for entry in self.database:
            text = f"Crop: {entry['crop']}, Disease: {entry['disease']}, Symptoms: {entry['symptoms']}, Treatment: {entry['treatment']}"
            self.documents.append(text)
        
        # Create embeddings
        self.doc_embeddings = self.embed_model.encode(self.documents)
        
        print(f"Created embeddings for {len(self.documents)} documents")
        print(f"Embedding dimension: {self.doc_embeddings.shape[1]}")
        print("Setup complete!")
    
    def search(self, query, top_k=3):
        """
        Search for relevant entries using cosine similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: Top matching entries with similarity scores
        """
        # Create query embedding
        query_embedding = self.embed_model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            entry = self.database[idx]
            result = {
                'crop': entry['crop'],
                'disease': entry['disease'],
                'symptoms': entry['symptoms'],
                'treatment': entry['treatment'],
                'similarity_score': similarities[idx],
                'index': idx
            }
            results.append(result)
        
        return results
    
    def get_recommendation(self, query, top_k=3):
        """Get formatted disease diagnosis and treatment recommendations."""
        results = self.search(query, top_k)
        
        if not results:
            return "No matching diseases found in the database."
        
        recommendation = f"Based on your query: '{query}'\n\n"
        recommendation += "Top recommendations:\n\n"
        
        for i, result in enumerate(results, 1):
            recommendation += f"{i}. **{result['crop']} - {result['disease']}**\n"
            recommendation += f"   Symptoms: {result['symptoms']}\n"
            recommendation += f"   Treatment: {result['treatment']}\n"
            recommendation += f"   Similarity: {result['similarity_score']:.3f}\n\n"
        
        return recommendation
