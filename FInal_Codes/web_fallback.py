from tavily import TavilyClient
import trafilatura
from sentence_transformers import SentenceTransformer
import numpy as np
from llama_cpp import Llama
from config import Settings

# -------------------------
# CONFIG
# -------------------------
MAX_DOC_CHARS = 1200        # per document
MAX_TOTAL_CHARS = 3000      # total context (SAFE for 4k ctx)
SIM_THRESHOLD = 0.10

settings = Settings()
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)


class WebFallbackService:
    def __init__(self, model_path: str):
        self.embedding_model = SentenceTransformer("all-miniLM-L6-v2")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0
        )

    # -------------------------
    # Web search + scrape
    # -------------------------
    def web_search(self, query: str):
        print("\n🔍 [WEB SEARCH TRIGGERED]")
        print("🔎 Query:", query)

        results = []
        response = tavily_client.search(query, max_results=5)

        for r in response.get("results", []):
            url = r.get("url")
            print("🌐 URL:", url)

            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print("⬇️ Downloaded HTML: NO")
                continue

            content = trafilatura.extract(downloaded, include_comments=False)
            if not content or len(content) < 200:
                print("⚠️ Skipped: insufficient content")
                continue

            content = content[:MAX_DOC_CHARS]

            print("📄 Extracted length:", len(content))

            results.append({
                "title": r.get("title", ""),
                "url": url,
                "content": content
            })

        print("✅ Total usable web documents:", len(results))
        return results

    # -------------------------
    # Sort by relevance
    # -------------------------
    def sort_sources(self, query: str, docs: list[dict]):
        print("\n🧠 [SORTING SOURCES]")
        relevant = []

        q_emb = self.embedding_model.encode(query)

        for d in docs:
            emb = self.embedding_model.encode(d["content"])
            sim = np.dot(q_emb, emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(emb)
            )

            print(f"📊 Similarity {sim:.3f} → {d['url']}")

            if sim >= SIM_THRESHOLD:
                d["score"] = sim
                relevant.append(d)

        relevant.sort(key=lambda x: x["score"], reverse=True)
        print("✅ Relevant documents kept:", len(relevant))
        return relevant

    # -------------------------
    # Final answer
    # -------------------------
    def answer(self, query: str) -> str:
        print("\n🧪 [WEB FALLBACK ANSWER PIPELINE]")

        search_results = self.web_search(query)
        ranked = self.sort_sources(query, search_results)

        if not ranked:
            return "இணையத்தில் தொடர்புடைய தகவல் கிடைக்கவில்லை."

        # 🔐 SAFE CONTEXT BUILD
        context_parts = []
        total_chars = 0

        for i, r in enumerate(ranked):
            block = f"Source {i+1}:\n{r['content']}\n"
            if total_chars + len(block) > MAX_TOTAL_CHARS:
                break
            context_parts.append(block)
            total_chars += len(block)

        context = "\n".join(context_parts)

        print("📚 Total context size:", total_chars, "chars")

        prompt = f"""
நீங்கள் ஒரு அறிவார்ந்த உதவியாளர்.

கீழே இணையத்தில் இருந்து பெறப்பட்ட தகவல்கள் உள்ளன.
இந்த தகவல்களை வைத்து கேள்விக்கு நேரடியாக பதிலளிக்கவும்.

❌ சுருக்கம் செய்ய வேண்டாம்
❌ Instruction / Response என்று எழுத வேண்டாம்
✅ இறுதி பதிலை மட்டும் வழங்கவும்

தகவல்கள்:
{context}

கேள்வி:
{query}

பதில்:
"""

        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            stop=["</s>"]
        )

        answer = output["choices"][0]["text"].strip()
        print("✅ [WEB RAG] Answer generated")
        return answer
