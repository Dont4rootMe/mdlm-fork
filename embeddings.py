import torch
import transformers
import torch.nn.functional as F
import hashlib
import os
import pickle
import numpy as np

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEmbedder:
    """
    A class to embed a batch of texts using an embedding model (e.g., NVEmbed).
    The output embedding dimension is available as self.cond_dim.
    """

    def __init__(self, model_name, device="cuda", random_projection_dim=None):
        """
        Args:
            model_name: Name or path of the embedding model (e.g., "NVEmbed", "sentence-transformers/all-MiniLM-L6-v2").
            device: Device to run the model on.
            random_projection_dim: If provided, applies fixed random projection to this dimension (non-learnable).
        """

        self.device = device
        self.random_projection_dim = random_projection_dim

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

        # Get original embedding dimension first
        dummy_text = ["dummy"]
        original_outputs = self._compute_embeddings(dummy_text)
        self.original_dim = original_outputs.shape[-1]
        
        # Initialize random projection matrix if needed
        self.random_projection_matrix = None
        if random_projection_dim is not None:
            # Create fixed random projection matrix (Gaussian random projection)
            # No bias, as requested
            np.random.seed(42)  # Fixed seed for reproducibility
            self.random_projection_matrix = torch.tensor(
                np.random.normal(0, 1 / np.sqrt(self.original_dim), 
                               (self.original_dim, random_projection_dim)),
                dtype=original_outputs.dtype, device=device
            )
            self.cond_dim = random_projection_dim
        else:
            self.cond_dim = self.original_dim

    @torch.no_grad()
    def __call__(self, texts):
        """
        Args:
            texts: List of strings (batch of texts).
        Returns:
            embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        """
        embeddings = self._compute_embeddings(texts)
        
        # Apply random projection if configured
        if self.random_projection_matrix is not None:
            embeddings = torch.matmul(embeddings, self.random_projection_matrix)
        
        return embeddings

    def _compute_embeddings(self, texts):
        """
        Compute embeddings for a batch of texts using the underlying model.
        Args:
            texts: List of strings.
        Returns:
            List or tensor of embeddings.
        """
        # Try to use .encode() method first (for sentence-transformers models)
        if hasattr(self.model, 'encode'):
            try:
                return self.model.encode(texts)
            except Exception as e:
                print(f"Failed to use .encode() method: {e}. Falling back to manual computation.")
        
        # Fall back to manual tokenization and mean pooling
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        assert sentence_embeddings.shape[0] == len(texts)
        assert len(sentence_embeddings.shape) == 2

        return sentence_embeddings


class TextEmbedderWithCache(TextEmbedder):
    """
    A TextEmbedder with caching functionality.
    Inherits from TextEmbedder and adds caching for computed embeddings.
    """

    def __init__(self, model_name, device="cuda", cache_path="/home/shibaiv/diff/mdlm/cache/text_embeddigs/", random_projection_dim=None):
        """
        Args:
            model_name: Name or path of the embedding model (e.g., "NVEmbed", "sentence-transformers/all-MiniLM-L6-v2").
            device: Device to run the model on.
            cache_path: Path to store cached embeddings.
            random_projection_dim: If provided, applies fixed random projection to this dimension (non-learnable).
        """
        # Initialize the parent class
        super().__init__(model_name, device, random_projection_dim)
        
        # Add caching functionality
        self.cache_path = cache_path
        self.cache_file = os.path.join(cache_path, "embeddings_cache.pkl")
        
        # Initialize cache
        self.cache = {}
        self.cache_additions = 0
        self.cache_save_threshold = 1000
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_path, exist_ok=True)
        
        # Load existing cache if it exists
        self._load_cache()

    def _get_text_hash(self, text):
        """Generate a hash for a text string."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _load_cache(self):
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache)} cached embeddings to {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")

    @torch.no_grad()
    def __call__(self, texts):
        """
        Args:
            texts: List of strings (batch of texts).
        Returns:
            embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        """
        # Check which texts are already cached
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                cached_embeddings.append((i, self.cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute embeddings for uncached texts
        new_outputs = None
        if uncached_texts:
            new_outputs = self._compute_embeddings(uncached_texts)
            
            # Cache the new embeddings (original embeddings, before projection)
            for text, embedding in zip(uncached_texts, new_outputs):
                text_hash = self._get_text_hash(text)
                self.cache[text_hash] = embedding
                self.cache_additions += 1
            
            # Save cache if threshold reached
            if self.cache_additions >= self.cache_save_threshold:
                self._save_cache()
                self.cache_additions = 0
        
        # Combine cached and newly computed embeddings in the correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place newly computed embeddings
        if uncached_texts and new_outputs is not None:
            for i, embedding in enumerate(new_outputs):
                original_idx = uncached_indices[i]
                all_embeddings[original_idx] = embedding
        
        # Convert to tensor and stack
        outputs = torch.stack([torch.tensor(embedding) if not isinstance(embedding, torch.Tensor) else embedding for embedding in all_embeddings])
        
        # Apply random projection if configured
        if self.random_projection_matrix is not None:
            outputs = torch.matmul(outputs, self.random_projection_matrix)
        
        return outputs
