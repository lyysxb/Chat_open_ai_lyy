from langchain.document_loaders import DirectoryLoader
from langchain.chains.conversation.base import ConversationChain
from langchain.chains import LLMChain
import os
import re
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage
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
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
f_r = open("E:\games\Chat_open_ai\prompt.txt","r",encoding = "utf-8")
f_w = open(r"E:\games\Chat_open_ai\train.txt","a+",encoding = "utf-8")
chat = ChatOpenAI(temperature=0)
prompt = ""
for line in f_r.readlines():
    prompt = prompt + line

for i in tqdm(range(100)):
    train_txt = chat([HumanMessage(content=prompt)]).content
    f_w.write(train_txt)
