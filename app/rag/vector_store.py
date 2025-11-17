"""
Faiss-based Vector Store for RAG
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI

from app.core.config import get_settings

settings = get_settings()


class FaissVectorStore:
    """Faiss 기반 벡터 저장소"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or settings.RAG_STORAGE_PATH)
        self.storage_path.mkdir(exist_ok=True)
        
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner Product (코사인 유사도)
        self.dimension = 1536  # text-embedding-3-small 차원
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # 기존 데이터 로드
        self._load()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """OpenAI로 임베딩 생성"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # L2 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding
    
    def add_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]]
    ):
        """문서 추가"""
        if len(documents) != len(metadatas):
            raise ValueError("documents와 metadatas 길이가 다릅니다")
        
        # Faiss 인덱스 초기화
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # 임베딩 생성
        embeddings = []
        for doc in documents:
            emb = self._get_embedding(doc)
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        
        # Faiss에 추가
        self.index.add(embeddings)
        
        # 메타데이터 저장
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        # 저장
        self._save()
    
    def search(
        self, 
        query: str, 
        n_results: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """유사도 검색"""
        if not self.documents or self.index is None:
            return {
                "documents": [],
                "metadatas": [],
                "distances": []
            }
        
        # 쿼리 임베딩
        query_embedding = self._get_embedding(query).reshape(1, -1)
        
        # Faiss 검색
        distances, indices = self.index.search(query_embedding, min(n_results * 3, len(self.documents)))
        
        # 필터링
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Faiss는 결과가 없으면 -1 반환
                continue
            
            # 메타데이터 필터
            if filter_metadata:
                if not all(
                    self.metadatas[idx].get(k) == v 
                    for k, v in filter_metadata.items()
                ):
                    continue
            
            results.append((idx, dist))
            
            if len(results) >= n_results:
                break
        
        return {
            "documents": [self.documents[idx] for idx, _ in results],
            "metadatas": [self.metadatas[idx] for idx, _ in results],
            "distances": [float(dist) for _, dist in results]
        }
    
    def _save(self):
        """데이터 저장"""
        # Faiss 인덱스 저장
        if self.index is not None:
            faiss.write_index(self.index, str(self.storage_path / "faiss.index"))
        
        # 메타데이터 저장
        data = {
            "documents": self.documents,
            "metadatas": self.metadatas
        }
        with open(self.storage_path / "metadata.pkl", "wb") as f:
            pickle.dump(data, f)
    
    def _load(self):
        """데이터 로드"""
        index_path = self.storage_path / "faiss.index"
        metadata_path = self.storage_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            # Faiss 인덱스 로드
            self.index = faiss.read_index(str(index_path))
            
            # 메타데이터 로드
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]
    
    def clear(self):
        """모든 데이터 삭제"""
        self.documents = []
        self.metadatas = []
        self.index = faiss.IndexFlatIP(self.dimension)
        self._save()
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension
        }


# 싱글톤 인스턴스
_vector_store: Optional[FaissVectorStore] = None


def get_vector_store() -> FaissVectorStore:
    """벡터 스토어 싱글톤"""
    global _vector_store
    if _vector_store is None:
        _vector_store = FaissVectorStore()
    return _vector_store
