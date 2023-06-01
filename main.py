import chromadb
from chromadb.config import Settings
from langchain.document_loaders import DirectoryLoader
from langchain.chains.conversation.base import ConversationChain
from langchain.chains import LLMChain
import os
import re
import openai
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import json
from Chat_open_ai.update_embedding import first_encode,update_encode

os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_KEY'
user = "lyy"
api_key = "YOUR_OPENAI_KEY"
def load_all_courses(solidity_root):
  loader = DirectoryLoader(solidity_root,glob="*/*/*.txt")#,glob=**/*.txt
  docs = loader.load()
  return docs

embeddings = OpenAIEmbeddings(openai_api_key = api_key)
path = r"./Konwledge/"
docs = load_all_courses(path)
print("使用的知识文档数目:" + str(len(docs)))
print("使用的第一个文档页数:" + str(len(docs[0].page_content)))

persist_directory = f'{user}_chroma_storage'
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
split_docs=text_splitter.split_documents(docs)
print("进行分割过后的知识文档数目" + str(len(split_docs)))
if os.path.exists(persist_directory) == False:
  print("正在为知识文档创建向量数据库")
  vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
  vectorstore.persist()
  retriever = vectorstore.as_retriever()
else:
  print("知识文档向量数据库已存在")
  vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
  retriever = vectordb.as_retriever()

# retriever = vectordb.as_retriever()
# print(vectordb)
# query="能给我介绍以下篮球的信息吗"
# docs=vectordb.similarity_search(query)
# print(len(docs))
# print(docs[0])
# 构造prompt
k = 1 #找到相似的文档 k 是可选参数,这里选1
# query="能给我介绍一下篮球这项运动的信息吗？并且给你自己的回复进行打分,并且解释你这么打分的原因"
# docs=vectordb.similarity_search(query,k,include_metadata=True)  #在这里做相似查询
system_template = """
结合context回答用户的问题.
如果你不知道答案，回答unkonw，不要编造答案.，并且用中文回答.
-----------
{context}
-----------
{chat_history}
"""
system2_template = """
根据下面的query和answer对answer进行打分并且分值在1-10间，只需要给出分数.
-----------
query:{query}
-----------
answer:{answer}
"""

# system3_template =  """
# 根据下面的query进行文本分类，一共有6类，分别是体育历史:1,体育品牌:2,体育意义:3,体育技巧与规则,体育明星,体育赛事，比如：我想了解足球运动的规则，分类为.
# -----------
# query:{query}
# -----------
# answer:{answer}
# """
# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template('{question}')
]
#用来做做人机对话

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)

# 初始化问答链
#可能需要把对话模型化成Huggingface上的模型
model = ChatOpenAI(temperature=0.1,max_tokens=2048) #语言模型可改变
qachain1 = ConversationalRetrievalChain.from_llm(model,retriever,combine_docs_chain_kwargs={'prompt':prompt},return_source_documents=True) #用来做检索式的对话
qachain2 = LLMChain(llm=ChatOpenAI(temperature=0),prompt=PromptTemplate.from_template(system2_template)) #用来对对话打分，必须使用openai
chat_history = []
if os.path.exists(user):
  print("用户历史聊天总窗口已存在")
else:
  os.mkdir(user)
  print("正在为用户创建总历史聊天窗口")
id = 0 #创建问题对编号
# topic = "" #初始化话题
while True:
  question = input('问题：')
  #####################################用来进行意图分类的代码###############################
  # if "足球" in question:
  #   topic = "足球"
  # if "游泳" in question:
  #   topic = "游泳"
  # if "乒乓球" in question:
  #   topic = "乒乓球"
  # if "马拉松" in question:
  #   topic = "马拉松"
  # if "篮球" in question:
  #   topic = "篮球"
  # persist_directory = f'./{user}_{topic}_chroma_storage'
  # if os.path.exists(persist_directory) == False:
  #   print(f"正在为{topic}知识文档创建向量数据库")
  #   vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
  #   vectorstore.persist()
  #   retriever = vectorstore.as_retriever()
  # else:
  #   print(f"{topic}知识文档向量数据库已存在")
  #   vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
  #   retriever = vectordb.as_retriever()

  # 初始化问答链
  # 可能需要把对话模型化成Huggingface上的模型
  # qachain1 = ConversationalRetrievalChain.from_llm(model, retriever, combine_docs_chain_kwargs={'prompt': prompt},return_source_documents=True)  # 用来做检索式的对话
  # qachain2 = LLMChain(llm=ChatOpenAI(temperature=0),prompt=PromptTemplate.from_template(system2_template))  # 用来对对话打分，必须使用openai
  # qachain3 = LLMChain(llm=ChatOpenAI(temperature=0), prompt=PromptTemplate.from_template(system2_template))
  ########################################################################
  if question == "结束":
    break
  # 开始发送问题 chat_history 为必须参数,用于存储对话历史
  result = qachain1({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print("AI回复:" + result['answer'])
  print(result)
  print(result['source_documents'][0])
  score = qachain2.run({"query":question,'answer':result['answer']})
  pattern = re.search(r'\d+', score)
  score = int(pattern.group())
  print("本次回复AI判定得分:" + str(score))
  # 拿到所有的question进行比较，选取相似度最高的三个，返回对应的问答
  # 下面应该保存为json格式
  if score > 7 :
    if os.path.exists(f"././{user}/{user}.jsonl"):
      data = {
        "question": question,
        "answer": result["answer"],
        "score": str(score),
        "id": id
      }
      print("正在连接问答数据库,存储问答对")
      f_r = open(f"./{user}/{user}.jsonl", "r", encoding="utf-8")
      print("正在为您找到你的最后一个问题id，方便存储")
      for line in f_r.readlines():
        id_json = json.loads(line)  # 找到最后一个id
        id = id_json["id"]
      data["id"] = int(id) + 1

      with open(f"./{user}/{user}.jsonl", "a+", encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        print("存储问答对成功")
        file.write("\n")
    else:
      print("似乎没有创建问答数据库，正在创建问答数据库")
      os.mkdir(f"./{user}_query_storage")
      with open(f"./{user}/{user}.jsonl","a+",encoding='utf-8') as file:
        data = {
          "question": question,
          "answer": result["answer"],
          "score": str(score),
          "id": str(id)
        }
        json.dump(data, file,ensure_ascii = False)
        file.write("\n")
    if os.path.exists(f"./{user}_query_storage"):
      print("用户问答向量数据库已存在,直接加载最新query")
      update_encode(question,id,score,api_key,user)
    else:
      os.mkdir(f"./{user}_query_storage")
      print("正在为用户创建问答向量数据库,并加载所有query")
      first_encode(f"./{user}/{user}.jsonl",api_key,user)
  if int(score) < 7:
    print("本次得分过低，将从数据库历史查找相似问题")
    if os.path.exists(f"././{user}/{user}.jsonl"):
      print("正在再次连接问答数据库")
    else:
      print("似乎没有创建问答数据库")
      break
    chroma_client = chromadb.Client(Settings(
      chroma_db_impl="duckdb+parquet",
      persist_directory=f"./{user}_query_storage"  # Optional, defaults to .chromadb/ in the current directory
    ))
    collection = chroma_client.get_collection(name=user + "collection")
    embedding = openai.Embedding.create(input = question, model="text-similarity-ada-001")['data'][0]['embedding']
    results = collection.query(
      query_embeddings = embedding,
      n_results=1
    )
    print(results)











