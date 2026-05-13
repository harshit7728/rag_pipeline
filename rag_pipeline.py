import chromadb
import json
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#define here your api key
api_key=""
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
                page_content="Harshit Panchal is a backend developer specializing in Python, Django, FastAPI, and scalable API architectures. He has experience working with DevOps tools such as Docker, Kubernetes, Jenkins, Terraform, and AWS cloud services including EC2, S3, RDS, and EKS. Harshit has also worked on Retrieval-Augmented Generation (RAG) systems using vector databases like ChromaDB and embedding models from Gemini and OpenAI. His projects include document-based question answering systems, automated CI/CD deployment pipelines, and distributed microservices applications. In addition to backend engineering, he is interested in AI engineering, asynchronous programming, Redis caching, and production-level system design."
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
    dataset = ["""Load balancing is a key component of high-availability systems. It distributes incoming network traffic across multiple servers to ensure no single server becomes a bottleneck.
        "Horizontal scaling involves adding more machines to your resource pool, whereas vertical scaling means adding more power (CPU, RAM) to an existing machine.
        "System peak load is handled by an auto-scaling group that monitors CPU utilization and triggers new instance spinning when thresholds are exceeded.
        "The caching layer, using Redis or Memcached, significantly reduces latency by storing frequently accessed data in memory, avoiding expensive database lookups.
        "Database sharding is the process of storing a large database across multiple machines. This helps in scaling the data layer and improving query performance under heavy load.
        "Microservices architecture allows teams to deploy services independently. Each service handles a specific business function and communicates via lightweight APIs.
        "Continuous Integration and Continuous Deployment (CI/CD) pipelines automate the testing and deployment of code, ensuring rapid and reliable software releases.
        "Health checks are vital for monitoring system status. If a service instance fails its health check, the load balancer stops routing traffic to it until it recovers.
        "Rate limiting prevents abuse of APIs by restricting the number of requests a user can make within a certain timeframe.
        "Read replicas improve database performance by offloading read traffic from the primary instance, allowing it to focus on write operations.""",]


    pipeline = RAGPipeline()
    pipeline.ingest(dataset)

    queries = [
        "How does the system handle peak load?",
        "Ways to scale the database",
        "Reducing latency and response times"
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