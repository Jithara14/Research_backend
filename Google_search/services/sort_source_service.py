
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class SortSourceService :
    def __init__(self):
        self.embedding_model =  SentenceTransformer("all-miniLM-L6-v2") 

    def sort_sources(self , query:str , search_results:List[dict]):
        relevent_docs = []
        query_embedding = self.embedding_model.encode(query)

        for res in search_results:
            content = res.get("content")

            # Skip if content is None or empty
            if content is None or str(content).strip() == "":
                #print("⚠️ Skipped empty content:", res)
                continue

            # encode only valid content
            res_embedding = self.embedding_model.encode(content)

            similarity = np.dot(query_embedding, res_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(res_embedding)
            )

            res["relevant_Score"] = similarity

            if similarity > 0.15:
                relevent_docs.append(res)
            
        return sorted(relevent_docs, key=lambda x: x["relevant_Score"], reverse=True)

