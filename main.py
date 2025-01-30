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
            Extract THE SINGLE JOB POSTING into JSON with keys: 
            `role`, `experience`, `skills` (as array), and `description`.
            ### VALID JSON (NO MARKDOWN):
        """)
        
        chain_extract = extract_prompt | llama
        res = chain_extract.invoke(input={'page_data': page_scrap})
        
        # Handle potential array response
        json_res = JsonOutputParser().parse(res.content)
        if isinstance(json_res, list):
            json_res = json_res[0]  # Take first job if multiple
            
        # Ensure skills is a list
        skills = json_res.get('skills', [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(',')]
        
        # Query portfolio database
        results = collection.query(
            query_texts=skills,
            n_results=2
        )
        
        # Extract unique links from results
        links = []
        if results and 'metadatas' in results:
            for metadata_list in results['metadatas']:
                for item in metadata_list:
                    if 'links' in item:
                        links.append(item['links'])
        unique_links = list(set(links))[:3]  # Get top 3 unique links
        
        # Generate email
        email_prompt = PromptTemplate.from_template("""
            ### JOB DESCRIPTION:
            {job_data}
            
            ### RELEVANT WORK LINKS:
            {links}
            
            ### INSTRUCTION:
            Write a professional email showing how our experience matches 
            these requirements. Use the links to showcase relevant work.
            ### EMAIL (NO PREAMBLE):
        """)
        
        chain_email = email_prompt | llama
        email_res = chain_email.invoke({
            "job_data": str(json_res),
            "links": "\n- ".join(unique_links)
        })
        
        return {"email": email_res.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)