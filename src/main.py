# -*- coding: utf-8 -*-
import requests
import pandas as pd
from io import StringIO
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import prompts
import sparql_queries


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


def store_dataset_information_to_chroma_db(chroma_path: str, embedding: OpenAIEmbeddings) -> None:
    """
    Stores the embedding vectors of concatenated titles and descriptions for all datasets accessible via the SPARQL API
    in a Chroma database, along with basic dataset metadata (title, URI, and description).
    :param chroma_path: Path to the ChromaDB folder.
    :param embedding: The embedding function.
    :return:
    """

    # get basic dataset information and store in a dataframe
    x = requests.get(f'http://statistics.gov.scot/sparql.csv', params={'query': sparql_queries.get_all_datasets_and_info})
    df = pd.read_csv(StringIO(x.text), sep=",")

    # insert into the chroma vector database
    df['to_vec'] = df['label'] + ': ' + df['desc']
    docs = []
    for i, j in df.iterrows():
        doc = Document(
            page_content=j['to_vec'],
            metadata={"ds_title": j['label'], "ds_link": j['ds'], "ds_description": j['desc']}
        )
        docs.append(doc)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=chroma_path
    )

    vectordb.persist()


def chatqa(query: str, model: str) -> str:
    """
    Basic OpenAI inference through langchain.
    :param query: The user's query.
    :param model: The name of the OpenAI model to be used.
    :return: The model's output
    """

    llm = ChatOpenAI(model_name=model, temperature=0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template="""{q}""", input_variables=["q"]))
    res = chain.apply([{'q': query}])[0]['text']

    return res


def reply_over_a_dataset(question: str,  chroma_directory: str = 'data/datasets_chroma', refresh_datasets_chroma: bool = False) -> str:
    """
    Given a user's question that can be answered through API-accessible information, this function uses an LLM to
    construct and execute API queries to the endpoint, retrieving and presenting the relevant details back to the user.

    :param question: The user's question.
    :param refresh_datasets_chroma: Whether the chroma store containing information about the datasets should be
    refreshed (True) or not (False).
    :param chroma_directory: The path to the ChromaDB directory.
    :return: The model's response.
    """

    # setup dataset chroma store
    embedding = OpenAIEmbeddings()
    if not os.path.exists(chroma_directory) or refresh_datasets_chroma:
        store_dataset_information_to_chroma_db(chroma_path=chroma_directory, embedding=embedding)
    ds_vectordb = Chroma(persist_directory=chroma_directory, embedding_function=embedding)

    # retrieve relevant dataset uri
    ds_retriever = ds_vectordb.as_retriever(search_kwargs={"k": 1})
    relevant_dataset = ds_retriever.get_relevant_documents(question)[0]

    # add question information to main prompt
    prompt = prompts.main_prompt.substitute(question=question)

    # add dataset information to main prompt
    prompt += f'Dataset {relevant_dataset.metadata["ds_title"]} with the DATASET_URI <{relevant_dataset.metadata["ds_link"]}>\n'

    # get dataset dimensions
    x = requests.get(f'http://statistics.gov.scot/sparql.csv', params={'query': sparql_queries.get_all_properties.substitute(dataset=relevant_dataset.metadata['ds_link'])})
    dims = pd.read_csv(StringIO(x.text), sep=",", dtype=str)
    measure_cl = dims[dims['dim'] == 'http://purl.org/linked-data/cube#measureType']['cl'].values[0]

    # get measures dataset
    x = requests.get(f'http://statistics.gov.scot/sparql.csv', params={'query': sparql_queries.get_measures.substitute(measure_cl=measure_cl)})
    meass = pd.read_csv(StringIO(x.text), sep=",", dtype=str)
    dims = dims[dims['dim'] != 'http://purl.org/linked-data/cube#measureType']

    dims_and_measures_response = chatqa(
        prompts.dims_and_measures_narrowing_prompt.substitute(question=question, d_labels=', '.join(dims['label'].tolist()), m_labels=', '.join(meass['label'].tolist())),
        model='gpt-3.5-turbo-0613'
    )
    dims_to_be_used_labels, measures_to_be_used_labels = [[j.strip() for j in i.split(',')] for i in dims_and_measures_response.splitlines()]

    dims_to_be_used = dims[dims['label'].isin(dims_to_be_used_labels)]
    dims_to_be_used_urls = dims_to_be_used['dim'].tolist()
    dims_to_be_used_codelists_urls = dims_to_be_used['cl'].tolist()

    measures_to_be_used = meass[meass['label'].isin(measures_to_be_used_labels)]
    measures_to_be_used_urls = measures_to_be_used['measure'].tolist()

    for mea, mea_l in zip(measures_to_be_used_urls, measures_to_be_used['label'].tolist()):
        prompt += f"The measure {mea_l}, with the MEASURE_URI <{mea}>\n"

    for cl, dim, dim_l in zip(dims_to_be_used_codelists_urls, dims_to_be_used_urls, dims_to_be_used['label'].tolist()):
        # get codelists for every measure selected to be used
        x = requests.get(f'http://statistics.gov.scot/sparql.csv', params={'query': sparql_queries.get_cls.substitute(cl=cl)})
        clvals_and_names = pd.read_csv(StringIO(x.text), sep=",", dtype=str)

        cl1_response = chatqa(
            prompts.codelist_narrowing_prompt.substitute(question=question, dim_l=dim_l, dim_vs=','.join(clvals_and_names['name'].tolist())),
            model='gpt-3.5-turbo-0613'
        )
        cl1_response = [i.strip() for i in cl1_response.split(',')]
        cls_vals_to_be_used = clvals_and_names[clvals_and_names['name'].isin(cl1_response)]['clvals'].tolist()
        prompt += f"The dimension {dim_l}, with the DIMENSION_URI <{dim}> can use the DIMENSION_VALUE_URIs: {', '.join([f'<{i[1]}> denoting {i[0]}' for i in list(zip(cl1_response, cls_vals_to_be_used))])}\n"

    constructed_query = chatqa(prompt, model='gpt-3.5-turbo-16k-0613')
    x = requests.get(f'http://statistics.gov.scot/sparql.csv', params={'query': constructed_query})

    final_q = prompts.final_response_prompt.substitute(question=question, sparql_final_result=" ".join(x.text.splitlines()))
    final_answer = chatqa(final_q, model='gpt-3.5-turbo-0613')

    return final_answer
