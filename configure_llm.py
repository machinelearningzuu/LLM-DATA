import yaml, os, openai
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.embeddings import (
                                    HuggingFaceEmbedding, 
                                    OpenAIEmbedding
                                    )
from llama_index.node_parser import SentenceSplitter
from llama_index import (
                        SimpleDirectoryReader,
                        set_global_service_context,
                        VectorStoreIndex
                        )
from constants import *

os.chdir(working_dir)

with open('cadentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

llm_model = credentials['OPENAI_GPT4_ENGINE'] if gpt_flag == 'GPT4' else credentials['OPENAI_GPT3_ENGINE']

if embedding_flag == 'OPENAI':
    embedding_llm = OpenAIEmbedding(
                                model="text-embedding-3-small"
                                )

else:
    embedding_llm = HuggingFaceEmbedding(
                                        model_name="BAAI/bge-small-en-v1.5",
                                        device='mps'
                                        )

chat_llm = OpenAI(
                api_key=credentials['OPENAI_API_KEY'],
                model=llm_model,
                temperature=0.3
                )

completion_llm = OpenAI(
                        api_key=credentials['OPENAI_API_KEY'],
                        model=llm_model,
                        temperature=0
                        )

audio_client = openai.OpenAI(
                    api_key=credentials['OPENAI_API_KEY']
                )

node_parser = SentenceSplitter(chunk_size=1024)

service_context = ServiceContext.from_defaults(
                                            embed_model=embedding_llm,
                                            node_parser=node_parser,
                                            llm=chat_llm
                                            )

set_global_service_context(service_context)
