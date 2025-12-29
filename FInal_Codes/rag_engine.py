from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp


class ArticleRAG:
    def __init__(self, text_file_path: str, model_path: str):
        # 1️⃣ Load article text
        loader = TextLoader(text_file_path, encoding="utf-8")
        docs = loader.load()

        # 2️⃣ Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

       
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
        )


        # 4️⃣ Vector DB
        self.db = FAISS.from_documents(chunks, embeddings)

        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.6
            }
        )

        # 5️⃣ LLM (same as your code)
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.2,
            max_tokens=256,
            n_ctx=4096,
            verbose=False
        )

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)

        if not docs:
            return (
                "நீங்கள் கேட்ட கேள்விக்கான பதில் கொடுக்கப்பட்ட தகவல்களில் இல்லை இணையத்தில் தேட வேண்டுமா?"
            )

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
கீழே வழங்கப்பட்டுள்ள 'செய்தி' தவிர வேறு எந்த அறிவையும் பயன்படுத்த வேண்டாம்.
கேள்விக்கான பதில் 'செய்தி' இல் இல்லை என்றால்,
'எனக்கு தெரியவில்லை' என்று மட்டும் கூறவும்.

செய்தி:
{context}

கேள்வி: {question}
"""

        raw = self.llm.invoke(prompt)
        return raw.replace(prompt, "").strip()
