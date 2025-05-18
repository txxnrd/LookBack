# rag/wiki_live_chain.py
from langchain_community.document_loaders import WikipediaLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

def build_wiki_live_chain(model_name: str, device: int = 0):
    llm = HuggingFacePipeline(model_name=model_name, device=device)

    emb_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

    def wiki_retriever(query: str):
        # ① 위키 문단 3개 받아오기
        docs = WikipediaLoader(query=query,
                               lang="en",
                               load_max_docs=3).load_and_split()

        # ② on-the-fly 벡터 DB (문단 수가 작아 매우 빠름)
        store = FAISS.from_documents(docs, emb_model)
        return store.similarity_search(query, k=4)

    prompt = PromptTemplate(
        template="{context}\n\n#Question#: {question}\n#Answer#:",
        input_variables=["context", "question"],
    )

    return RetrievalQA(llm=llm,
                       retriever=wiki_retriever,
                       prompt=prompt)
