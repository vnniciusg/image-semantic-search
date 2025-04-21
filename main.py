import os 
import requests
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field

__import__('warnings').filterwarnings('ignore')

from PIL import Image
from tqdm import tqdm
from loguru import logger
from matplotlib import pyplot as plt
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class ImageStore(BaseModel):
    path: str = Field(..., description="Path to the image")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the image")
    embedding: Optional[list[float]] = Field(None, description="Optional precomputed embedding of the image")


class OpenCLIPSemanticSearch():

    def __init__(self) -> None:
        self.embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint='laion2b_s34b_b88k')
        self.image_store: list[ImageStore] = []

    def add_image(self, image_uri: str, metadata: Optional[dict] = None) -> None:
        if not os.path.exists(image_uri):
            raise FileNotFoundError(f'image not found: {image_uri}')
        
        img_embedding = self.embeddings.embed_image([image_uri])[0]
        self.image_store.append(ImageStore(path=image_uri, metadata=metadata, embedding=img_embedding))

    def search(self, query: str, top_k: int = 5, score_threshold: Optional[float] = None) -> list[dict]:
        try:
            if isinstance(query, str) and os.path.exists(query):
                query_embedding = self.embeddings.embed_image([query])[0]
            elif isinstance(query, str):
                query_embedding = self.embeddings.embed_documents([query])[0]
            
            image_results = []
            if self.image_store:            
                similarities = [cosine_similarity(np.array([query_embedding]), np.array([img.embedding]))[0][0] for img in self.image_store]
                
                sorted_indices = np.argsort(similarities)[::-1]
                for idx in sorted_indices[:top_k]:
                    score = similarities[idx]
                    if score_threshold is None or score >= score_threshold:
                        image_results.append({
                            'path': self.image_store[idx].path, 
                            'score': round(float(score), 4),
                            'metadata': self.image_store[idx].metadata
                        })
            
            return image_results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
        
def get_images():
    os.makedirs('images', exist_ok=True)

    image_urls = [
        'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/7/71/Calico_tabby_cat_-_Savannah.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/1/15/Cat_August_2010-4.jpg'
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for i, url in tqdm(enumerate(image_urls), desc='Downloading images', total=len(image_urls)):
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(f'images/image_{i}.jpg', 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def display_results(results):
    if not results:
        print("No results found")
        return
    
    num_results = len(results)
    cols = min(3, num_results)
    rows = (num_results + cols - 1) // cols
    
    plt.figure(figsize=(15, 5*rows))
    
    for i, result in enumerate(results, 1):
        plt.subplot(rows, cols, i)
        img = Image.open(result['path'])
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Score: {result['score']:.4f}")
    
    plt.tight_layout()
    plt.show()

def main():

    if not os.path.exists('images') or len(os.listdir('images')) == 0:
        get_images()

    semantic_search = OpenCLIPSemanticSearch()


    logger.info("adding images to the semantic search")
    for i, filename in enumerate(os.listdir('images')):
        img_path = os.path.join('images', filename)
        if os.path.exists(img_path):
            semantic_search.add_image(img_path, {"index": i})

    query = 'a orange cat'
    logger.info(f"searching for: {query}")
    results = semantic_search.search(query, top_k=5)
    
    display_results(results)

if __name__ == '__main__':
    main()