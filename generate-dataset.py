import re, json
import pandas as pd
from typing import Tuple, List
from llama_index.llms import OpenAI
from llama_index.schema import BaseNode
from llama_index.prompts import (
                                ChatMessage,
                                MessageRole,
                                PromptTemplate,
                                ChatPromptTemplate,
                                )
from configure_llm import *

print(f"Use API from {llm_prefix}")
print(f"Use {gpt_flag} as LLM")
print(f"Use E.LLM from {embedding_flag}")

documents = SimpleDirectoryReader(
                                input_dir='/home/external/TayshaFinetuning/LLM-DATA/biotech-docs'
                                ).load_data()
nodes = service_context.node_parser.get_nodes_from_documents(documents)
print(f"\n{nodes} nodes loaded\n")

QA_PROMPT = PromptTemplate(
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information and not prior knowledge, "
                        "answer the query.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                        )   

QUESTION_GEN_USER_TMPL = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information and not prior knowledge, "
                        "generate the relevant questions. "
                        )

QUESTION_GEN_SYS_TMPL = """\
You are a Medical Professional about Gene Therapies. Your task is to setup \
{num_questions_per_chunk} questions for which assumes that patients who involving Gene Therapies \
about to ask from Doctors who specialized Gene Therapies . The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.\
"""

question_gen_template = ChatPromptTemplate(
                                        message_templates=[
                                                        ChatMessage(role=MessageRole.SYSTEM, content=QUESTION_GEN_SYS_TMPL),
                                                        ChatMessage(role=MessageRole.USER, content=QUESTION_GEN_USER_TMPL),
                                                        ]
                    )

def generate_answers_for_questions(
                                    questions: List[str], 
                                    context: str, 
                                    llm: OpenAI
                                    ) -> str:
    answers = []
    for question in questions:
        fmt_qa_prompt = QA_PROMPT.format(
                                        context_str=context, 
                                        query_str=question
                                        )
        response_obj = llm.complete(fmt_qa_prompt)
        answers.append(str(response_obj))
    return answers

def generate_qa_pairs(
                    llm: OpenAI, 
                    nodes: List[BaseNode], 
                    num_questions_per_chunk: int = 10
                    ) -> List[Tuple[str, str]]:
    
    if not os.path.exists('generated/biotech'):
        os.makedirs('generated/biotech')
    else:
        if len(os.listdir('generated/biotech')) > 0:
            for file in os.listdir('generated/biotech'):
                os.remove(os.path.join('generated/biotech', file))
            
    qa_pairs = []
    for idx, node in enumerate(nodes):
        print(f"processing node {idx}/{len(nodes)}")
        context_str = node.get_content(metadata_mode="all")
        fmt_messages = question_gen_template.format_messages(
                                                            num_questions_per_chunk=num_questions_per_chunk,
                                                            context_str=context_str,
                                                            )
        chat_response = llm.chat(fmt_messages)
        raw_output = chat_response.message.content
        result_list = str(raw_output).strip().split("\n")
        cleaned_questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip()
            for question in result_list
        ]
        answers = generate_answers_for_questions(
                                                cleaned_questions, 
                                                context_str, 
                                                llm
                                            )
        for q, a in zip(cleaned_questions, answers):
            qa_pairs.append({
                            "question": q,
                            "answer": a,
                            "context": context_str
                            })
            
        with open(f'generated/biotech/qa_{idx}.json', 'w') as f:
            json.dump(qa_pairs, f)

generate_qa_pairs(
                service_context.llm, nodes, 
                num_questions_per_chunk=2
                )
