import chromadb
import json
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#define here your api key
api_key="AIzaSyCb-0h3OfkB84VUrGbw_V2EuBrTv6gBygY"
class EnhancedQuery(BaseModel):
    query:str=Field(description="expand the query")

class RAGPipeline:
    def __init__(self):
      
        
        self.gen_model= ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",api_key=api_key
            )
        self.vectore_store=Chroma(
                collection_name="example_collection",
                embedding_function=GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            ),
                persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
    def ingest(self,text_recived):
        docs=[]
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # texts = text_splitter.split_text(text_recived)
        print("texts",text_recived)
        for text in text_recived:

            doc = Document(
                page_content=text
            )
            docs.append(doc)
        if docs:
            self.vectore_store.add_documents(docs)

    def retrieve_strategy_a(self,query):
        result=self.vectore_store.similarity_search(query,k=3)
        response=[]
        if result:
            for res in result:
                response.append(res.page_content)
        return response
        
    def retrieve_strategy_b(self,query):
        strct_query=self.gen_model.with_structured_output(EnhancedQuery)
        expansion_prompt = f"Rewrite this query for better semantic retrieval: {query}"
        response=strct_query.invoke(expansion_prompt).model_dump()
        print("response",response)
        expanded_query=response['query']
        result=self.vectore_store.similarity_search(query=expanded_query,k=3)
        response=[]
        if result:
            for res in result:
                response.append(res.page_content)

        return response,expanded_query
        


def run_benchmark():
    dataset = [
    "Artificial Intelligence is transforming the healthcare industry in many ways. Hospitals use AI-powered systems to detect diseases earlier and improve diagnosis accuracy.",

    "Cloud computing allows companies and individuals to store and access data over the internet instead of relying on local computers.",

    "Regular exercise is essential for maintaining good physical and mental health and helps reduce stress and anxiety.",

    "E-commerce has changed the way people buy and sell products through online shopping platforms.",

    "Renewable energy comes from natural sources such as solar, wind, and hydroelectric power."
]

    pipeline = RAGPipeline()
    pipeline.ingest(dataset)

    queries = [
    "What is the main idea of the first paragraph?",
    "How does technology improve modern life according to the paragraphs?",
    "Why are health and environmental topics important?"
]

    report = []
    for q in queries:
        results_a = pipeline.retrieve_strategy_a(q)
        results_b, expanded = pipeline.retrieve_strategy_b(q)
        
        report.append({
            "original_query": q,
            "expanded_query": expanded,
            "strategy_a_results": results_a,
            "strategy_b_results": results_b
        })
    
    return report


if __name__ == "__main__":
   
    results = run_benchmark()
    print("rresults",results)
    print("############################### \n",json.dumps(results, indent=2))