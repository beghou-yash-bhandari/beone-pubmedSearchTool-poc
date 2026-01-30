import os
import requests
from langchain_openai import ChatOpenAI
import pandas as pd
import time
import xml.dom.minidom as minidom
import xmltodict
   
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,JsonOutputParser
import urllib.parse
import json
from typing import List, Dict, Any, Optional
import json
from Bio import Entrez
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import psycopg2
from langchain_openai.embeddings import OpenAIEmbeddings


# loading the OpenAI api key from .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large" , dimensions=1536)
def pretty_print_xml(xml_string):
    data = xmltodict.parse(xml_string)
    return json.dumps(data, indent=2)

PINECONE_INDEX_US = os.getenv("PINECONE_INDEX_US")
PINECONE_API_KEY_US = os.getenv("PINECONE_API_KEY")

# pc_us = Pinecone(api_key=PINECONE_API_KEY_US)
# index_us = pc_us.Index(PINECONE_INDEX_US)
# store_US = PineconeVectorStore(index=index_us, embedding=embeddings, namespace='PSV-V1-BETA')


# from langchain_text_splitters import RecursiveCharacterTextSplitter
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor
os.environ["OPENAI_API_KEY"] = "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"


 
# Setting up prompts
df = pd.read_excel("Prompt_Design_PubMed_Parameters_With_Prompts.xlsx")
rate_chart = df[df['Allowed'] == 1][['Data_num','Data']]
rate_chart_dict = rate_chart.set_index('Data_num')['Data'].to_dict()
print(rate_chart_dict)

prompt_map = (
    df.assign(Data_num=df["Data_num"].astype(str).str.strip())
      .set_index("Data_num")["LLM Extraction Prompt"]
      .to_dict()
)
def load_prompt_assets(excel_path: str):
    """
    Returns:

      - rate_chart_dict: {Data_num -> Data} for Allowed==1
      - prompt_map: {Data_num -> LLM Extraction Prompt}
    """
    df = pd.read_excel(excel_path)

    #Pushkal

    rate_chart = df[df["Allowed"] == 1][["Data_num", "Data"]]
    rate_chart_dict = (
        rate_chart.assign(Data_num=rate_chart["Data_num"].astype(str).str.strip())
        .set_index("Data_num")["Data"]
        .to_dict()
    )

    rate_chart_dict_op = (
        rate_chart.assign(Data_num=rate_chart["Data_num"].astype(str).str.strip())
        .set_index("Data")["Data_num"]
        .to_dict()
    )

    prompt_map = (
        df.assign(Data_num=df["Data_num"].astype(str).str.strip())
        .set_index("Data_num")["LLM Extraction Prompt"]
        .to_dict()
    )

    print("Loaded rate_chart_dict keys:", list(rate_chart_dict.keys())[:10], "...")
    print("Loaded prompt_map keys:", list(prompt_map.keys())[:10], "...")
    return rate_chart_dict, prompt_map, rate_chart_dict_op
# Helping Func


def extract_pubmed_document(data):
    """
    Convert PubMed XML (from Entrez.read) into a LangChain Document.
    `data` can be:
      - the whole response with key 'PubmedArticleSet', or
      - a single 'PubmedArticle' dict.
    """
    
    # ----------------------------------
    # Normalize to a single PubmedArticle
    # ----------------------------------
    if "PubmedArticleSet" in data:
        pas = data["PubmedArticleSet"]
        # can be list or single dict
        pubmed_article = pas.get("PubmedArticle", {})
    else:
        pubmed_article = data.get("PubmedArticle", data)

    medline = pubmed_article.get("MedlineCitation", {})
    article = medline.get("Article", {})

    # -------------------------
    # PMID
    # -------------------------
    pmid = ""
    pmid_raw = medline.get("PMID", "")
    if isinstance(pmid_raw, dict):
        pmid = pmid_raw.get("#text", "")
    else:
        pmid = str(pmid_raw) if pmid_raw else ""

    # -------------------------
    # DOI
    # -------------------------
    doi = ""
    eloc = article.get("ELocationID", {})
    # ELocationID can be dict or list of dict
    eloc_list = []
    if isinstance(eloc, list):
        eloc_list = eloc
    elif isinstance(eloc, dict):
        eloc_list = [eloc]

    for e in eloc_list:
        if e.get("@EIdType", "").lower() == "doi":
            doi = e.get("#text", "")
            break

    # -------------------------
    # Title
    # -------------------------
    title = article.get("ArticleTitle", "")
    if isinstance(title, dict):
        title = title.get("#text", "")
    if not isinstance(title, str):
        title = str(title)

    # -------------------------
    # Abstract
    # -------------------------
    abstract = ""
    abstract_block = article.get("Abstract", {})
    if isinstance(abstract_block, dict):
        at = abstract_block.get("AbstractText", "")
        if isinstance(at, str):
            abstract = at
        elif isinstance(at, list):
            # Each element may be dict or str
            parts = []
            for x in at:
                if isinstance(x, dict):
                    parts.append(x.get("#text", ""))
                else:
                    parts.append(str(x))
            abstract = " ".join(part for part in parts if part)
        elif isinstance(at, dict):
            abstract = at.get("#text", "")
        else:
            abstract = str(at)

    # -------------------------
    # Authors + Affiliations
    # -------------------------
    authors = []
    affiliations_set = set()

    author_list_block = article.get("AuthorList", {})
    # AuthorList is a dict with 'Author' key
    authors_raw = author_list_block.get("Author", [])
    if isinstance(authors_raw, dict):
        authors_raw = [authors_raw]

    for a in authors_raw:
        last = a.get("LastName", "") or ""
        fore = a.get("ForeName", "") or ""
        full_name = f"{fore} {last}".strip()
        if full_name:
            authors.append(full_name)

        # AffiliationInfo may be list or dict
        aff_info = a.get("AffiliationInfo", [])
        if isinstance(aff_info, dict):
            aff_info = [aff_info]
        for aff in aff_info:
            aff_text = aff.get("Affiliation", "")
            if aff_text:
                affiliations_set.add(aff_text)

    affiliations_str = "; ".join(sorted(affiliations_set)) if affiliations_set else ""

    # -------------------------
    # Journal + Pub date
    # -------------------------
    journal_block = article.get("Journal", {})
    journal_title = journal_block.get("Title", "")
    if isinstance(journal_title, dict):
        journal_title = journal_title.get("#text", "")
    if not isinstance(journal_title, str):
        journal_title = str(journal_title)

    pub_year = pub_month = pub_day = ""
    journal_issue = journal_block.get("JournalIssue", {})
    pub_date = journal_issue.get("PubDate", {})

    if isinstance(pub_date, dict):
        pub_year = str(pub_date.get("Year", "") or "")
        pub_month = str(pub_date.get("Month", "") or "")
        pub_day = str(pub_date.get("Day", "") or "")
        if not pub_year and "MedlineDate" in pub_date:
            medline_date = pub_date["MedlineDate"]
            if isinstance(medline_date, str) and len(medline_date) >= 4:
                pub_year = medline_date[:4]
    else:
        pub_year = str(pub_date) if pub_date else ""

    # Build an ISO-ish date string but do not force missing parts
    pub_parts = [p for p in [pub_year, pub_month, pub_day] if p]
    publication_date = "-".join(pub_parts)

    # -------------------------
    # Keywords / MeSH
    # -------------------------
    keywords = []

    # MeSH terms
    mesh_block = medline.get("MeshHeadingList", {})
    mesh_items = mesh_block.get("MeshHeading", [])
    if isinstance(mesh_items, dict):
        mesh_items = [mesh_items]

    for mesh in mesh_items:
        descriptor = mesh.get("DescriptorName")
        if isinstance(descriptor, dict):
            kw = descriptor.get("#text", "")
        else:
            kw = str(descriptor) if descriptor else ""
        if kw:
            keywords.append(kw)

    # Deduplicate + clean
    keywords = sorted(set(kw.strip() for kw in keywords if kw and kw.strip()))

    # -------------------------
    # URL
    # -------------------------
    url = f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}" if pmid else ""

    # -------------------------
    # Build main text block
    # -------------------------
    text_block = f"""
# {title}

## Journal
{journal_title}

## Authors
{", ".join(authors)}

## Publication Date
{publication_date}

## Keywords / MeSH
{", ".join(keywords)}

## Abstract
{abstract}

## Affiliations
{affiliations_str}

## Source
{url}
    """.strip()

    # -------------------------
    # Metadata
    # -------------------------
    metadata = {
        "title": title,
        "pmid": pmid,
        "doi": doi,
        "journal": journal_title,
        "authors": authors,
        "publication_date": publication_date,
        "keywords": keywords,
        "url": url,
        "affiliations": affiliations_str,
    }

    return Document(page_content=text_block, metadata=metadata)

def id_retriever(search_string, page=10):
    string = "+".join(search_string.split(' '))
    result = []
    dataEmptFlag = False
    old_result = len(result)
    # for page in range(1,5):
    req = requests.get("https://pubmed.ncbi.nlm.nih.gov/?term={string}&format=pmid&filter=simsearch2.ffrft&page=1&size={page}".format(string=string, page=page))
    # req = requests.get("https://pubmed.ncbi.nlm.nih.gov/?term={string}&format=pmid&sort=pubdate&size=200&page={page}".format(page=page,string=string))

    for j in req.iter_lines():
        decoded = j.decode("utf-8").strip(" ")
        length = len(decoded)
        if "log_displayeduids" in decoded and length > 46:
            data = (str(j).split('"')[-2].split(","))
            result = result + data

    return result

def ExtractArticles(q_broad,q_normal,q_narrow):
    Entrez.email = 'yash.bhandari@beghouconsulting.com'
    # handle = Entrez.esearch(db='pmc',RetMax=300, term=query, sort='relevance')
    # record = Entrez.read(handle)
    # print(record)
    # pmc_id_list = record['IdList']
    pm_id_list_broad=id_retriever(q_broad)
    print(len(pm_id_list_broad))
    print("pm_id_list_broad: ",pm_id_list_broad)
    pm_id_list_normal=id_retriever(q_normal)
    print(len(pm_id_list_normal))
    print("pm_id_list_normal: ",pm_id_list_normal)
    pm_id_list_narrow=id_retriever(q_narrow)
    print(len(pm_id_list_narrow))
    print("pm_id_list_narrow: ",pm_id_list_narrow)
    # pm_id_list = [pm_id_list[0] ] # just the first one for testing
    # res = pmid_to_pmcid(pm_id_list)
    # pmc_id_list = [v for v in res.values() if v is not None]
    pm_id_list = list(set(pm_id_list_broad[:20] + pm_id_list_normal[:20] + pm_id_list_narrow[:20]))

    print(len(pm_id_list))
    print("Final pm_id_list: ",pm_id_list)


    if not pm_id_list:
        raise ValueError("No PMC IDs found for this query")

    documents = []
    # pmc_id_list = pmc_id_list[:100]
    for pmc_id in pm_id_list:

        # ---------- FETCH ARTICLE XML ----------
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=pmc_id,
            retmode="xml"
        )
        raw_xml = fetch_handle.read()
        fetch_handle.close()

        # (Optional) pretty print just for inspection
        # pretty_xml = pretty_print_xml(raw_xml)
        # print(pretty_xml)

        # ---------- PARSE XML TO DICT ----------
        data = xmltodict.parse(raw_xml)
        # print(data)
    
        doc = extract_pubmed_document(data)
        documents.append(doc)
        time.sleep(0.33)

    my_list = documents

    with open(r"output\document_retd.txt", "w" , encoding="utf-8") as f:
        for item in my_list:
            f.write(f"item###################################################################\n {item.page_content}\n")
    return documents

def EnhancedQuery(query,model = "gpt-5.2"):
    
    model = ChatOpenAI(model = model)

    prompt_1 = f"""You are an expert search strategist for clinical literature databases such as PubMed. Your task is to transform a simple, plain-text user query into a single Enhanced, Structured Query that improves clarity and organization while preserving the full meaning and breadth of the original query.

    Your enhancements must be additive only—never restrictive, never substitutive.

    Core Enhancement Rules (Strict and Required)
    1. Preserve Meaning Exactly

    You must not reinterpret, replace, or redirect the original clinical concepts in the user’s input.
    The enhanced query must retain the same conceptual focus as the user’s original wording.

    2. Additive-Only Enhancement

    You may enhance the original query by:

    Adding a MeSH term that directly corresponds to a disease, drug, or intervention explicitly mentioned by the user.

    Adding closely related synonyms or phrase variants only when they reinforce the same concept already present.
    You must not remove or weaken any original search terms.
    You must not introduce new conceptual categories, such as epidemiology terms, natural history, diagnostics, mechanistic biology, or treatment modalities not present in the original query.

    3. Controlled Synonym Expansion

    For any concept in the user’s query:

    Include the exact original term(s).

    Add synonyms only if they are natural linguistic variants of the same concept.

    Do not expand into adjacent or broader clinical concepts.

    Example:
    If the user mentions “line of progression,” you may add synonyms related to treatment sequencing, but you may not introduce unrelated terms like “disease course” or “natural history.”

    All expanded terms must be joined with OR.

    4. MeSH Terms (Light, Only When Directly Relevant)

    Add a MeSH term only if:

    The user clearly expresses a disease, drug class, or therapeutic agent that matches an official MeSH descriptor.

    Do not add MeSH terms for concepts the user did not explicitly mention.

    5. Boolean Structure

    Combine major concept groups with AND.

    Combine synonyms within a concept using OR.

    Use parentheses to maintain clear structure.

    6. Field Tags (Minimal Use)

    Use:

    [MeSH] only for MeSH terms

    [Title/Abstract] only if user specificly asked for this

    Avoid excessive tagging that restricts sensitivity.

    7. Maintain Search Sensitivity

    The enhanced query must remain at least as broad as the user’s original intent.
    If an enhancement risks narrowing the results, you must omit that enhancement.

    Output Formatting Rule (Strict):

    You must output the final Enhanced, Structured Query as one single line of plain text with no line breaks, no newline characters (\n), no code block formatting, no backticks, and no surrounding quotes.

    Do not introduce any escape characters or special formatting.
    The output must be a clean PubMed-compatible Boolean query exactly as it should be pasted into the PubMed search bar.

    Input Query:

    {query}"""
    # response = model.invoke(f"Extract all the key mesh terms / keywords from the user query. User Query: {query}.Return only the terms comma separated")
    prompt_2_broad = f"""
    Make a good and broad pubmed search query. User wants to research on topic: {query}. Do not give any other extra jargon, your answers will directly go in the search box so just give the query.
    """
    prompt_2_normal = f"""
    Make a good pubmed search query. User wants to research on topic: {query}. Do not give any other extra jargon, your answers will directly go in the search box so just give the query.
    """
    prompt_2_narrow = f"""
    Make a good and precise pubmed search query. User wants to research on topic: {query}. Do not give any other extra jargon, your answers will directly go in the search box so just give the query.
    """
    response_broad = model.invoke(prompt_2_broad)
    q_broad = response_broad.content
    response_normal = model.invoke(prompt_2_normal)
    q_normal = response_normal.content
    response_narrow = model.invoke(prompt_2_narrow)
    q_narrow = response_narrow.content
    return q_broad,q_normal,q_narrow

def split_large_documents(
    docs: List[Document],
    max_chars: int = 600_000,
) -> List[Document]:
    """
    Roughly enforce a size limit for each Document's page_content.
    For any document whose text is longer than `max_chars`, split its
    page_content into multiple Documents with identical metadata,
    plus a 'chunk_index' to distinguish them.

    Assumes ~3–4 chars/token and targets a 200k token context window
    with some safety margin.
    """
    new_docs: List[Document] = []

    for doc_idx, doc in enumerate(docs):
        text = doc.page_content or ""
        metadata = dict(doc.metadata or {})

        if len(text) <= max_chars:
            # Already small enough
            new_docs.append(doc)
            continue

        # Split into fixed-size character chunks
        num_chunks = (len(text) + max_chars - 1) // max_chars
        for chunk_idx in range(num_chunks):
            start = chunk_idx * max_chars
            end = start + max_chars
            chunk_text = text[start:end]

            chunk_metadata = {
                **metadata,
                "original_doc_index": doc_idx,
                "chunk_index": chunk_idx,
                "num_chunks": num_chunks,
            }

            new_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    return new_docs

def get_param_records(rate, document,model):

    # 1. Define schema for a single row
    class ParameterRecord(BaseModel):
        # param fix pushkal
        parameter: str = Field(description="Name of the parameter being extracted")
        extracted_value: str = Field(description="Value extracted for the parameter. Can not be unreported or none.Must have a valid value aligned with the given abstract")
        unit: Optional[str] = Field(
            default=None,
            description="Unit for the extracted value, if applicable"
        )
        population_and_subgroup: Optional[str] = Field(
            default=None,
            description="Population and subgroup details, if present"
        )
        study_context: Optional[str] = Field(
            default=None,
            description="Context such as baseline characteristic, efficacy endpoint, safety outcome"
        )
        exact_evidence_sentence: str = Field(
            description="Exact supporting sentence from the source text"
        )
        confidence_score: str = Field(
            description="How accurate is the extracted information in regards to the evidence provided. Give the score on the scale of 0-10, decimal are allowed."
        )
        resoning: str = Field(
            description="Resoning behind what ever information you extracted about the parameter."
        )

    # 2) Define schema for a list of records
    class ParameterRecords(BaseModel):
        # check pushkal
        records: List[Optional[ParameterRecord]] = Field(
            description="List of extracted parameter records. Return Empty list if you find nothing for the parameter"
        )

    # 5. Initialize LLM
    llm = ChatOpenAI(
        model=model
    )

    llm_with_structured_output = llm.with_structured_output(schema=ParameterRecords)

    rate_prompt = prompt_map[rate]

    prompt = f"""
    {rate_prompt}

    Note: You can fetch multiple readings too. I expect a list of records in a structured format. read the given Scientific literature/study Abstract carefully and extract all the posiable readings.

    Scientific literature/study Abstract:
    {document}
    """

    output = llm_with_structured_output.invoke(prompt)
    # Pushkal: For loop lagake, output structure dekhkar edit karlo

    # Pushkal : rate_name = rate_chart_dict[rate]; 
    # Param value rate name

    for record in output.records:
        if record is None:
            continue
        record.parameter = rate_chart_dict[rate]


    # print(rate_chart_dict[rate])
    return (document, output)

def ParsePubmed(documents, rates, model = 'gpt-5.2'):
    if 'all' in rates:
        rates = list(rate_chart_dict.keys())

    document_abstracts = split_large_documents(documents, max_chars=800_000)

    inputs = [(rate, abstract, model) for rate in rates for abstract in document_abstracts]
    print('Parsing ', len(inputs), ' Inputs')
    # inputs = inputs[:1]

    print('Parsing start..')
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: get_param_records(*p), inputs))

    all_rows: List[Dict[str, Any]] = []

    print('Formating results...')
    for doc, result in results:
        metadata = doc.metadata or {}
        rows = result.model_dump(exclude_none=True)
        for row in rows['records']:
            # Add metadata WITHOUT mutating the original row
            if not row:
                continue
            enriched = {**row, **metadata}
            all_rows.append(enriched)

    return pd.DataFrame(all_rows)

def SearchAndExtract(query,rate, model = "gpt-5.2"):

    q_broad,q_normal,q_narrow = EnhancedQuery(query)
    print('Enriched Query Broad:', q_broad)
    print('Enriched Query Normal:', q_normal)
    print('Enriched Query Narrow:', q_narrow)
    documents = ExtractArticles(q_broad,q_normal,q_narrow)
    print('Found ', len(documents), ' Documents')
    final_df = ParsePubmed(documents, rate, model)
    print('Sample Data:\n',final_df.sample())
    return final_df

 
# #1. User Input

# Hey_what_are_you_researching_about = "Line progression for CML"
# ALLOWED_RATE_TYPES = 'all'

 
# # 2. SearchAndExtract

# final_df = SearchAndExtract(query=Hey_what_are_you_researching_about, rate=ALLOWED_RATE_TYPES)

 

# # 3. Save to Excel

# final_df.to_excel(r"output\test2.xlsx", index=False)

# print(final_df)


