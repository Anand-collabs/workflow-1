from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import chromadb
import pandas as pd
import uuid

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

# Initialize components
llama = ChatGroq(
    api_key='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',
    model_name='llama-3.3-70b-versatile'
)

# Initialize ChromaDB
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

# Load portfolio data
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1rX3J2MJeKmASTtkScAo_6jgPukgpu9Be")
if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=row["Techstack"],
            metadatas={"links": row["Links"]},
            ids=[str(uuid.uuid4())]
        )

@app.get("/")
def root():
    """Root endpoint to indicate service is live."""
    return {"message": "CEG Backend Service is live!"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/generate-email")
async def generate_email(request: URLRequest):
    try:
        # Scrape job page
        loader = WebBaseLoader(request.url)
        page_scrap = loader.load().pop().page_content
        
        # Extract job details
        extract_prompt = PromptTemplate.from_template("""
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            Extract job postings into JSON with keys: `role`, `experience`, `skills` and `description`.
            ### VALID JSON (NO PREAMBLE):
        """)
        
        chain_extract = extract_prompt | llama
        res = chain_extract.invoke(input={'page_data': page_scrap})
        json_res = JsonOutputParser().parse(res.content)
        
        # Query portfolio database
        links = collection.query(
            query_texts=json_res['skills'], 
            n_results=2
        ).get('metadatas', [])
        
        # Generate email
        email_prompt = PromptTemplate.from_template("""
            ### JOB DESCRIPTION:
            {job_description}
            ### INSTRUCTION:
            Write a cold email as HR, BDE at Company, describing capabilities for this job.
            ### EMAIL (NO PREAMBLE):
        """)
        
        chain_email = email_prompt | llama
        email_res = chain_email.invoke({
            "job_description": str(json_res), 
            "link_list": links
        })
        
        return {"email": email_res.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)