import os
import faiss
import json
import openai
import chromadb
from chromadb.config import Settings


def first_encode(path,key,user):
    openai.api_key = key
    # if os.path.exists(f"./{user}_query_storage"):
    #     print("用户问答向量数据库已存在")
    # else:
    #     os.mkdir(f"./{user}_query_storage")
    #     print("正在为用户创建问答向量数据库")
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=f"./{user}_query_storage/"  # Optional, defaults to .chromadb/ in the current directory
    ))
    try:
        collection = chroma_client.create_collection(name=user + "collection")
    except ValueError:
        collection = chroma_client.get_collection(name = user + "collection")

    question_history = []
    scores = []
    ids = []
    embeddings = []
    f_r = open(path, "r", encoding="utf-8")

    for line in f_r.readlines():
        data = json.loads(line)
        question_history.append(data["question"])
        scores.append(int(data["score"]))
        ids.append(int(data["id"]))
    print(f"一共有{len(question_history)}正在被载入")
    for i in range(len(question_history)):
        embedding = openai.Embedding.create(input = question_history, model = "text-similarity-ada-001")['data'][i]['embedding']
        embeddings.append(embedding)
    # print(embeddings)
    collection.add(
        embeddings = embeddings,
        documents = question_history,
        metadatas = [{"score": scores[i]} for i in range(len(scores))],
        ids = [str(ids[i]) for i in range(len(ids))]
    )
    print("载入成功！")
    # print(collection.get(ids = ["4", "5", "6"]))

def update_encode(query,id,score,key,user):
    openai.api_key = key
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl = "duckdb+parquet",
        persist_directory = f"./{user}_query_storage/"  # Optional, defaults to .chromadb/ in the current directory
    ))
    collection = chroma_client.get_collection(name = user + "collection")
    embedding = openai.Embedding.create(input = query, model="text-similarity-ada-001")['data'][0]['embedding']
    print("一条query正在被载入")
    # print(collection.get(ids=["4", "5", "6"]))
    collection.add(
        embeddings = embedding,
        documents = [query],
        metadatas = [{"score": score}],
        ids = [str(id)]
    )
    print("载入成功！")
    # print(collection.get(ids = ["4", "5", "6"]))
# first_encode("E:\games\Chat_open_ai\lyy\lyy.jsonl",'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',"lyy")
# update_encode("你好啊",12,8,'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',"lyy")
# user = "lyy"
# chroma_client = chromadb.Client(Settings(
#         chroma_db_impl = "duckdb+parquet",
#         persist_directory=f"./{user}_query_storage"  # Optional, defaults to .chromadb/ in the current directory
#     ))
# collection = chroma_client.get_collection(name = user + "collection")
# openai.api_key =  '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'
# query_embedding =  openai.Embedding.create(input = "你好啊", model="text-similarity-ada-001")['data'][0]['embedding']
# results = collection.query(
#       query_embeddings = [query_embedding],
#       n_results = 2
#     )
# print(results)