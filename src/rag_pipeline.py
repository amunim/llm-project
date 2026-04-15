"""
RAG pipeline: ChromaDB retrieval + Qwen-2.5-3B-Instruct generation via LangChain.

Two LLM backends:
  - use_llama_cpp=True  → LlamaCpp (GGUF, CPU-compatible; for FastAPI / HF Spaces)
  - use_llama_cpp=False → HuggingFacePipeline (for Colab GPU)

Usage (standalone test):
    python src/rag_pipeline.py --question "What is the Little Champs Account?" \\
        --use-llama-cpp --model-path models/nust_bank_qwen2.5_3b_q4km.gguf
"""

import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """You are a helpful and professional customer service assistant for NUST Bank.
You answer questions about NUST Bank's products and services based ONLY on the provided context.

Rules:
- Only answer questions related to NUST Bank products and services.
- If the answer is not in the provided context, say "I don't have that information in our records. Please contact NUST Bank helpline at +92 (51) 111 000 494."
- Never provide financial advice. Only share verified information from bank documents.
- Be polite, professional, and concise.
- If someone asks a non-banking question, politely redirect them to NUST Bank services.

Context:
{context}

Customer Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["context", "question"])


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "bank_knowledge",
        max_new_tokens: int = 512,
        top_k: int = 5,
        use_llama_cpp: bool = False,
        model_path: str = "",
    ):
        # Embedding model (same one used during indexing)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        # LLM – two backends
        if use_llama_cpp:
            from langchain_community.llms import LlamaCpp
            if not model_path:
                raise ValueError("model_path is required when use_llama_cpp=True")
            self.llm = LlamaCpp(
                model_path=model_path,
                max_tokens=max_new_tokens,
                temperature=0.7,
                repeat_penalty=1.15,
                n_ctx=4096,
                n_batch=512,
                verbose=False,
            )
        else:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline as hf_pipeline,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            pipe = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.15,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

        # LangChain RetrievalQA chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        """Run a question through the RAG pipeline."""
        result = self.chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [
                {"text": doc.page_content[:200], **doc.metadata}
                for doc in result.get("source_documents", [])
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--db-path", default="data/chroma_db")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--use-llama-cpp", action="store_true")
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()

    rag = RAGPipeline(
        db_path=args.db_path,
        model_name=args.model,
        use_llama_cpp=args.use_llama_cpp,
        model_path=args.model_path,
    )
    result = rag.query(args.question)
    print("Answer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s}")
