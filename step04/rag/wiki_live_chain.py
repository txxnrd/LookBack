# rag/wiki_live_chain.py
from langchain_community.document_loaders import WikipediaLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline


def build_wiki_live_chain(
    model_name: str,
    device: int | str = 0,
    *,
    return_retriever_only: bool = False,   # ← 새 옵션
):
    """Wikipedia 실시간 검색 + LLM 파이프라인 생성

    Parameters
    ----------
    model_name : str
        🤗 허브 모델 이름 (ex. 'meta-llama/Llama-7b-chat-hf')
    device : int | str
        0, 1 … 또는 'cuda:0' 식 ID
    return_retriever_only : bool, default False
        • False → 기존과 동일: RetrievalQA 체인 리턴  
        • True  → (retriever, tokenizer) 튜플만 리턴
                  → 검색 결과만 쓰고 싶을 때 편리
    """
    # 1) LLM (generate 용)
    llm = HuggingFacePipeline(model_name=model_name, device=device)

    # 2) 문단 임베딩 모델
    emb_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

    # 3) on-the-fly 위키피디아 검색 함수
    def wiki_retriever(query: str):
        # (1) 위키 문단 3개 로드
        docs = WikipediaLoader(
            query=query,
            lang="en",
            load_max_docs=3,
        ).load_and_split()

        # (2) 문단 수가 작으니 즉석 벡터 DB 생성 → top-k=4 반환
        store = FAISS.from_documents(docs, emb_model)
        return store.similarity_search(query, k=4)

    # 4) RetrievalQA 프롬프트 (체인을 그대로 쓸 경우)
    prompt = PromptTemplate(
        template="{context}\n\n#Question#: {question}\n#Answer#:",
        input_variables=["context", "question"],
    )

    # 5) 체인 구성
    qa_chain = RetrievalQA(
        llm=llm,
        retriever=wiki_retriever,
        prompt=prompt,
    )

    # 6) 옵션에 따라 반환
    if return_retriever_only:
        return qa_chain.retriever, llm.tokenizer
    return qa_chain
