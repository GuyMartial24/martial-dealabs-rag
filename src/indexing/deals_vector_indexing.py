import json
import random
import time
import os
import glob
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class MongoDBConnectionError(Exception):
    pass


class EmbeddingError(Exception):
    pass


@dataclass
class Config:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dims: int = 384
    mongo_uri: str = os.getenv("MONGO_URI", "")
    db_name: str = "deals_db"
    collection_name: str = "deals"
    index_name: str = "vector_index"
    data_dir: str = str(PROJECT_ROOT / "data" / "newdeals")
    batch_size: int = 50


class EmbeddingService:
    """Gère la génération d'embeddings via SentenceTransformer."""
    
    def __init__(self, model_id: str, embedding_dims: int):
        self.embedding_dims = embedding_dims
        print(f"Chargement du modèle {model_id}...")
        self.model = SentenceTransformer(model_id)
        print("Modèle chargé")
    
    def embed(self, text: str) -> list[float]:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Echec de l'embedding: {e}")


class MongoDBClient:
    """Gère la connexion et les opérations MongoDB."""
    
    def __init__(self, config: Config):
        self.config = config
        self._connect()
    
    def _connect(self):
        try:
            self.client = MongoClient(self.config.mongo_uri)
            self.db = self.client[self.config.db_name]
            self.collection = self.db[self.config.collection_name]
            self.client.admin.command('ping')
            print("Connecté à MongoDB Atlas")
        except Exception as e:
            raise MongoDBConnectionError(f"Impossible de se connecter: {e}")
    
    def reset_collection(self):
        if self.config.collection_name in self.db.list_collection_names():
            self.collection.drop()
            print(f"Collection '{self.config.collection_name}' supprimée")
        else:
            print(f"Collection '{self.config.collection_name}' inexistante, création...")
    
    def insert_batch(self, documents: list[dict]) -> tuple[int, int]:
        """Insère un batch de documents. Retourne (inserted, failed)."""
        if not documents:
            return 0, 0
        
        try:
            result = self.collection.insert_many(documents, ordered=False)
            return len(result.inserted_ids), 0
        except BulkWriteError as bwe:
            inserted = bwe.details['nInserted']
            return inserted, len(documents) - inserted
        except Exception as e:
            print(f"Erreur batch: {e}")
            return 0, len(documents)
    
    def count_documents(self) -> int:
        return self.collection.count_documents({})
    
    def create_vector_index(self):
        print(f"Connexion établie à {self.config.db_name}.{self.config.collection_name}")
        
        try:
            self.collection.drop_search_index(self.config.index_name)
            print(f"Ancien index '{self.config.index_name}' supprimé")
        except Exception:
            pass

        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": self.config.embedding_dims,
                    "similarity": "cosine"
                },
                {"type": "filter", "path": "group_display_summary"},
                {"type": "filter", "path": "price"}
            ]
        }

        search_index_model = SearchIndexModel(
            definition=index_definition,
            name=self.config.index_name,
            type="vectorSearch"
        )

        print("Création de l'index en cours...")
        result = self.collection.create_search_index(search_index_model)
        print(f"Index créé: {result}")
        print("L'indexation peut prendre quelques minutes.")


class DealProcessor:
    """Transforme les deals en documents MongoDB avec embeddings."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    @staticmethod
    def pick_best_comment(deal: dict) -> Optional[str]:
        comments = deal.get("comments", [])
        if not comments:
            return None

        max_likes = max(
            c.get("reaction_counters", {}).get("like", 0) for c in comments
        )
        best_comments = [
            c for c in comments
            if c.get("reaction_counters", {}).get("like", 0) == max_likes
        ]
        return random.choice(best_comments).get("content_unformatted")
    
    @staticmethod
    def parse_price(price) -> Optional[float]:
        if price is None:
            return None
        try:
            return float(price)
        except (ValueError, TypeError):
            return None
    
    def build_embedding_text(self, deal: dict) -> str:
        parts = []
        
        if title := deal.get("title"):
            parts.append(f"Titre : {title}.")
        if category := deal.get("group_display_summary"):
            parts.append(f"Catégorie : {category}.")
        if description := deal.get("html_stripped_description"):
            parts.append(f"Description : {description[:500]}.")
        if (price := deal.get("price")) is not None:
            parts.append(f"Prix : {price} euros.")
        if temp := deal.get("temperature_level"):
            parts.append(f"Niveau d'attractivité : {temp}.")
        if best_comment := self.pick_best_comment(deal):
            parts.append(f"Commentaire populaire : {best_comment[:200]}.")
        
        return " ".join(parts)
    
    def prepare_document(self, deal_id: str, deal: dict) -> Optional[dict]:
        text = self.build_embedding_text(deal)
        
        try:
            embedding = self.embedding_service.embed(text)
        except EmbeddingError:
            return None
        
        if len(embedding) != self.embedding_service.embedding_dims:
            return None

        return {
            "_id": deal_id,
            "title": deal.get("title"),
            "text": text,
            "html_stripped_description": deal.get("html_stripped_description"),
            "group_display_summary": deal.get("group_display_summary"),
            "price": self.parse_price(deal.get("price")),
            "url": deal.get("deal_uri"),
            "embedding": embedding
        }


class DataLoader:
    """Charge les deals depuis des fichiers JSON."""
    
    @staticmethod
    def load_all_deals(data_dir: str) -> dict:
        all_deals = {}
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        print(f"Fichiers trouvés: {len(json_files)}")
        
        for filepath in sorted(json_files):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    
                    if isinstance(file_data, list):
                        for i, deal in enumerate(file_data):
                            deal_id = deal.get("deal_id") or deal.get("id") or f"{filename}_{i}"
                            all_deals[str(deal_id)] = deal
                    elif isinstance(file_data, dict):
                        all_deals.update(file_data)
                        
                print(f"  {filename} chargé")
            except json.JSONDecodeError as e:
                print(f"  Erreur JSON dans {filename}: {e}")
            except IOError as e:
                print(f"  Erreur lecture {filename}: {e}")
        
        return all_deals


class MigrationPipeline:
    """Orchestre la migration complète des deals vers MongoDB Atlas."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_service = EmbeddingService(config.model_id, config.embedding_dims)
        self.mongo_client = MongoDBClient(config)
        self.deal_processor = DealProcessor(self.embedding_service)
    
    def run(self, reset: bool = True, create_index: bool = True):
        print("Migration vers MongoDB Atlas")
        
        print(f"\nChargement des fichiers depuis {self.config.data_dir}...")
        data = DataLoader.load_all_deals(self.config.data_dir)
        print(f"Total: {len(data)} deals chargés")
        
        if reset:
            print("\nPréparation de la base de données...")
            self.mongo_client.reset_collection()
        
        print("\nInsertion en cours...")
        start_time = time.time()
        indexed, failed = self._index_deals(data)
        elapsed = time.time() - start_time
        
        print("\nRésumé:")
        print(f"  Documents insérés: {indexed}")
        print(f"  Echecs: {failed}")
        print(f"  Temps total: {elapsed:.1f}s")
        
        count = self.mongo_client.count_documents()
        print(f"\nTotal dans MongoDB Atlas: {count} documents")
        
        if create_index:
            print("\nCréation de l'index vectoriel...")
            self.mongo_client.create_vector_index()
    
    def _index_deals(self, data: dict) -> tuple[int, int]:
        total = len(data)
        indexed = 0
        failed = 0
        deals_list = list(data.items())
        
        for i in range(0, total, self.config.batch_size):
            batch = deals_list[i:i + self.config.batch_size]
            documents = []
            
            for deal_id, deal in batch:
                doc = self.deal_processor.prepare_document(deal_id, deal)
                if doc:
                    documents.append(doc)
                else:
                    failed += 1
            
            batch_indexed, batch_failed = self.mongo_client.insert_batch(documents)
            indexed += batch_indexed
            failed += batch_failed
            
            progress = min(i + self.config.batch_size, total)
            print(f"\rProgression: {progress}/{total} ({100 * progress // total}%)", end="", flush=True)
        
        print()
        return indexed, failed


if __name__ == "__main__":
    config = Config()
    pipeline = MigrationPipeline(config)
    pipeline.run()