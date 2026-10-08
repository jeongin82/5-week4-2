#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 알라딘 베스트셀러 RAG
#
# 1. 환경변수 및 OpenAI Client
# 2. ChromaDB 초기화
# 3. 알라딘 웹 크롤링
# 4. 크롤링 데이터를 FAQ로 변환
# 5. Chunking
# 6. Embedding
# 7. ChromaDB 저장
# 8. 질문 Retrieval
# 9. Context 구성
# 10. 정확한 데이터 질문 처리
# 11. GPT 답변 생성
# ============================================================

import os
import time
import re

import requests
from bs4 import BeautifulSoup

import chromadb

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# 1. 환경변수 및 OpenAI Client
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. "
        ".env 파일을 확인하세요."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# 2. ChromaDB 초기화
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_db"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name="aladin_faq"
)


# ============================================================
# 3. 알라딘 웹 크롤링
# ============================================================

URL = "https://www.aladin.co.kr/shop/common/wbest.aspx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    )
}


def crawl_bestsellers():
    """
    알라딘 베스트셀러 1~3페이지를 크롤링한다.
    """

    params = {
        "BestType": "Bestseller",
        "BranchType": 1,
        "CID": 0,
        "page": 1,
        "cnt": 1000,
        "SortOrder": 1
    }

    rows = []

    for page in range(1, 4):

        params["page"] = page

        response = requests.get(
            URL,
            params=params,
            headers=HEADERS,
            timeout=10
        )

        print(
            f"{page}페이지 응답 코드: "
            f"{response.status_code}"
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        books = soup.select(".ss_book_box")

        for book in books:

            try:
                title_tag = book.select_one("a.bo3")
                price_tag = book.select_one("span.ss_p2")
                rating_tag = book.select_one("span.star_score")

                title = title_tag.text.strip()
                link = title_tag["href"]
                price = price_tag.text.strip()
                rating = rating_tag.text.strip()

                rows.append({
                    "title": title,
                    "link": link,
                    "price": price,
                    "rating": rating,
                    "rank": len(rows) + 1
                })

            except AttributeError:
                continue

        print(
            f"{page}페이지 완료, "
            f"지금까지 {len(rows)}권"
        )

        time.sleep(1)

    return rows


# ============================================================
# 4. 크롤링 데이터를 FAQ로 변환
# ============================================================

def create_faq(rows):
    """
    크롤링한 책 정보를 FAQ 형태의 텍스트로 변환한다.
    """

    faq_documents = []

    for book in rows:

        title = book["title"]
        rank = book["rank"]
        price = book["price"]
        rating = book["rating"]
        link = book["link"]

        faq = f"""
도서명: {title}
순위: {rank}

Q. {title}의 가격은 얼마인가?
A. {title}의 가격은 {price}이다.

Q. {title}의 별점은 몇 점인가?
A. {title}의 별점은 {rating}이다.

Q. {title}의 알라딘 링크는 무엇인가?
A. {title}의 알라딘 링크는 {link}이다.
""".strip()

        faq_documents.append(faq)

    return faq_documents


# ============================================================
# 5. Chunking
# ============================================================

def chunk_text(
    text,
    chunk_size=400,
    chunk_overlap=50
):
    """
    텍스트를 일정한 크기의 Chunk로 나눈다.

    수업의 build_vector_db.ipynb에서 사용한
    chunk_size=400, chunk_overlap=50 방식을 참고한다.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        start += (
            chunk_size -
            chunk_overlap
        )

    return chunks


# ============================================================
# 6. Embedding
# ============================================================

def get_embedding(text):
    """
    텍스트를 Embedding 벡터로 변환한다.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ============================================================
# 7. FAQ → Chunk → Embedding → ChromaDB
# ============================================================

def save_to_chroma(faq_documents):
    """
    FAQ를 Chunk로 나누고 Embedding한 뒤
    ChromaDB에 저장한다.
    """

    # 기존 데이터 삭제
    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

    total_chunks = 0

    for faq_index, faq in enumerate(
        faq_documents
    ):

        chunks = chunk_text(faq)

        for chunk_index, chunk in enumerate(
            chunks
        ):

            print(
                f"Embedding: "
                f"{faq_index + 1}/"
                f"{len(faq_documents)} "
                f"- Chunk "
                f"{chunk_index + 1}/"
                f"{len(chunks)}"
            )

            embedding = get_embedding(
                chunk
            )

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": "Aladin Bestseller",
                        "faq_index": faq_index
                    }
                ],
                ids=[
                    f"faq_{faq_index}_chunk_{chunk_index}"
                ]
            )

            total_chunks += 1

    print(
        f"\nChromaDB 저장 완료: "
        f"{total_chunks}개 Chunk"
    )


# ============================================================
# 8. Retrieval
# ============================================================

def retrieve(
    query,
    top_k=5
):
    """
    query를 Embedding하여
    ChromaDB에서 가장 유사한 문서를 검색한다.

    rag_chatbot.ipynb의 retrieve() 구조를 참고한다.
    """

    query_embedding = get_embedding(
        query
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# ============================================================
# 9. Context 구성
# ============================================================

def build_context(results):
    """
    검색된 문서들을 하나의 Context로 묶는다.

    rag_chatbot.ipynb의
    context_texts → context_str 구조를 참고한다.
    """

    found_docs = results["documents"][0]

    found_metadatas = results["metadatas"][0]

    context_texts = []

    for doc_text, metadata in zip(
        found_docs,
        found_metadatas
    ):

        context_texts.append(
            f"<<source: {metadata['source']}>>\n"
            f"{doc_text}"
        )

    context_str = "\n\n".join(
        context_texts
    )

    return context_str


# ============================================================
# 10. 숫자 데이터 처리용 함수
# ============================================================

def extract_price(price_text):
    """
    가격 문자열에서 숫자만 추출한다.

    예:
    '16,650원' → 16650
    """

    numbers = re.sub(
        r"[^0-9]",
        "",
        price_text
    )

    if not numbers:
        return 0

    return int(numbers)


def extract_rating(rating_text):
    """
    별점 문자열에서 숫자를 추출한다.

    예:
    '8.9' → 8.9
    """

    match = re.search(
        r"\d+(?:\.\d+)?",
        rating_text
    )

    if not match:
        return 0.0

    return float(
        match.group()
    )


# ============================================================
# 11. 정확한 데이터 질문 판단
# ============================================================

def answer_data_question(
    question,
    rows
):
    """
    순위, 가격, 별점처럼 정확한 비교가 필요한
    질문을 원본 크롤링 데이터에서 처리한다.

    해당 질문이 아니면 None을 반환한다.
    """

    question_lower = question.lower()

    # --------------------------------------------------------
    # 1. 가장 높은 순위 / 1위
    # --------------------------------------------------------

    if (
        "가장 높은 순위" in question
        or "1위" in question
        or "일등" in question
        or "최고 순위" in question
    ):

        book = min(
            rows,
            key=lambda x: x["rank"]
        )

        return (
            f"1위 도서는 「{book['title']}」입니다.\n"
            f"가격은 {book['price']}이며, "
            f"별점은 {book['rating']}입니다."
        )

    # --------------------------------------------------------
    # 2. 가장 낮은 순위
    # --------------------------------------------------------

    if (
        "가장 낮은 순위" in question
        or "최하위" in question
        or "마지막 순위" in question
    ):

        book = max(
            rows,
            key=lambda x: x["rank"]
        )

        return (
            f"가장 낮은 순위의 도서는 "
            f"「{book['title']}」이며 "
            f"{book['rank']}위입니다.\n"
            f"가격은 {book['price']}이며, "
            f"별점은 {book['rating']}입니다."
        )

    # --------------------------------------------------------
    # 3. 가장 비싼 책
    # --------------------------------------------------------

    if (
        "가장 비싼" in question
        or "가격이 가장 높은" in question
        or "가격이 제일 높은" in question
    ):

        book = max(
            rows,
            key=lambda x: extract_price(
                x["price"]
            )
        )

        return (
            f"가장 비싼 도서는 "
            f"「{book['title']}」입니다.\n"
            f"가격은 {book['price']}이며, "
            f"순위는 {book['rank']}위입니다."
        )

    # --------------------------------------------------------
    # 4. 가장 저렴한 책
    # --------------------------------------------------------

    if (
        "가장 저렴한" in question
        or "가장 싼" in question
        or "가격이 가장 낮은" in question
    ):

        book = min(
            rows,
            key=lambda x: extract_price(
                x["price"]
            )
        )

        return (
            f"가장 저렴한 도서는 "
            f"「{book['title']}」입니다.\n"
            f"가격은 {book['price']}이며, "
            f"순위는 {book['rank']}위입니다."
        )

    # --------------------------------------------------------
    # 5. 별점이 가장 높은 책
    # --------------------------------------------------------

    if (
        "별점이 가장 높은" in question
        or "평점이 가장 높은" in question
        or "별점이 제일 높은" in question
        or "평점이 제일 높은" in question
    ):

        book = max(
            rows,
            key=lambda x: extract_rating(
                x["rating"]
            )
        )

        return (
            f"별점이 가장 높은 도서는 "
            f"「{book['title']}」입니다.\n"
            f"별점은 {book['rating']}이며, "
            f"순위는 {book['rank']}위입니다."
        )

    # --------------------------------------------------------
    # 6. 특정 순위
    # --------------------------------------------------------

    rank_match = re.search(
        r"(\d+)\s*위",
        question
    )

    if rank_match:

        target_rank = int(
            rank_match.group(1)
        )

        for book in rows:

            if book["rank"] == target_rank:

                return (
                    f"{target_rank}위 도서는 "
                    f"「{book['title']}」입니다.\n"
                    f"가격은 {book['price']}이며, "
                    f"별점은 {book['rating']}입니다."
                )

    return None


# ============================================================
# 12. 특정 책의 정확한 정보 검색
# ============================================================

def find_book_by_title(
    question,
    rows
):
    """
    질문에 책 제목이 포함되어 있다면
    해당 책의 정확한 정보를 찾는다.
    """

    for book in rows:

        title = book["title"]

        if title in question:

            return (
                f"「{title}」의 정보입니다.\n"
                f"- 순위: {book['rank']}위\n"
                f"- 가격: {book['price']}\n"
                f"- 별점: {book['rating']}\n"
                f"- 알라딘 링크: {book['link']}"
            )

    return None


# ============================================================
# 13. GPT 답변 생성
# ============================================================

def generate_answer_with_context(
    query,
    results
):
    """
    검색된 문서를 Context로 구성하여
    GPT에게 최종 답변을 요청한다.

    rag_chatbot.ipynb의
    generate_answer_with_context() 구조를 참고한다.
    """

    context_str = build_context(
        results
    )

    system_prompt = """
당신은 알라딘 베스트셀러 FAQ 도우미입니다.

다음 원칙을 반드시 지키세요.

1. 반드시 제공된 문서 내용에 근거해서만 답변하세요.
2. 문서에 없는 내용을 추측하거나 만들어내지 마세요.
3. 검색된 문서에서 질문에 대한 정확한 정보를 찾을 수 없다면
   "관련 정보를 찾지 못했습니다."라고 답변하세요.
4. 순위, 가격, 별점 등의 숫자는 문서에 있는 값을 그대로 사용하세요.
5. 사용자가 한국어로 질문하면 한국어로 답변하세요.
6. 지나치게 장황하지 않게 간결하고 이해하기 쉽게 답변하세요.
"""

    user_prompt = f"""
아래는 알라딘 베스트셀러에서 검색된 문서입니다.

[검색된 문서]
{context_str}

[사용자 질문]
{query}

검색된 문서만 근거로 질문에 답변하세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# 14. 전체 RAG 실행
# ============================================================

def main():

    print("=" * 60)
    print("알라딘 베스트셀러 RAG")
    print("=" * 60)

    # --------------------------------------------------------
    # 1단계: 웹 크롤링
    # --------------------------------------------------------

    print(
        "\n[1단계] "
        "알라딘 베스트셀러 크롤링"
    )

    rows = crawl_bestsellers()

    print(
        f"\n총 {len(rows)}권의 책을 수집했습니다."
    )

    if not rows:

        print(
            "크롤링 결과가 없습니다."
        )

        return

    # --------------------------------------------------------
    # 2단계: FAQ 생성
    # --------------------------------------------------------

    print(
        "\n[2단계] "
        "크롤링 데이터를 FAQ로 변환"
    )

    faq_documents = create_faq(
        rows
    )

    print(
        f"총 {len(faq_documents)}개의 "
        f"FAQ 문서를 만들었습니다."
    )

    print("\n[FAQ 예시]")
    print(
        faq_documents[0]
    )

    # --------------------------------------------------------
    # 3단계: Embedding + ChromaDB
    # --------------------------------------------------------

    print(
        "\n[3단계] "
        "FAQ → Chunk → Embedding → ChromaDB"
    )

    save_to_chroma(
        faq_documents
    )

    # --------------------------------------------------------
    # 4단계: 질문
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "RAG 질문"
    )

    print(
        "=" * 60
    )

    while True:

        question = input(
            "\n질문을 입력하세요 "
            "(종료: quit): "
        ).strip()

        if question.lower() == "quit":

            print(
                "\n프로그램을 종료합니다."
            )

            break

        if not question:

            print(
                "질문을 입력해주세요."
            )

            continue

        # ----------------------------------------------------
        # 5단계: 정확한 데이터 질문 확인
        # ----------------------------------------------------

        exact_answer = answer_data_question(
            question,
            rows
        )

        if exact_answer:

            print(
                "\n[정확한 데이터 검색 결과]"
            )

            print(
                exact_answer
            )

            continue

        # ----------------------------------------------------
        # 6단계: 특정 책 제목 확인
        # ----------------------------------------------------

        book_answer = find_book_by_title(
            question,
            rows
        )

        if book_answer:

            print(
                "\n[도서 정보 검색 결과]"
            )

            print(
                book_answer
            )

            continue

        # ----------------------------------------------------
        # 7단계: Vector Retrieval
        # ----------------------------------------------------

        print(
            "\n[검색 중...]"
        )

        results = retrieve(
            question,
            top_k=5
        )

        found_docs = results[
            "documents"
        ][0]

        if not found_docs:

            print(
                "\n관련 문서를 찾지 못했습니다."
            )

            continue

        # ----------------------------------------------------
        # 8단계: 검색 결과 출력
        # ----------------------------------------------------

        print(
            "\n[검색된 FAQ]"
        )

        for i, doc in enumerate(
            found_docs,
            start=1
        ):

            print(
                f"\n--- 검색 결과 {i} ---"
            )

            print(
                doc
            )

        # ----------------------------------------------------
        # 9단계: Context + GPT
        # ----------------------------------------------------

        print(
            "\n[최종 답변 생성 중...]"
        )

        answer = generate_answer_with_context(
            question,
            results
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "최종 답변"
        )

        print(
            "=" * 60
        )

        print(
            answer
        )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()