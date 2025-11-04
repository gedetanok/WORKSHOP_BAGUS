
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
from pathlib import Path

load_dotenv()

# Initialize clients
embeddings_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chat_client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Setup ChromaDB with OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name='text-embedding-3-small'
)

chroma_client = chromadb.PersistentClient('./pdf_db')


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            return text
    except Exception:
        return None


def load_pdfs_from_path(folder_path):
    """Load all PDFs from given path"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    documents = []
    pdf_files = list(folder.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if text:
            documents.append({
                'filename': pdf_path.name,
                'content': text
            })
    
    return documents


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def load_documents_to_db(folder_path, collection):
    """Load PDFs and add to ChromaDB"""
    documents = load_pdfs_from_path(folder_path)
    
    if not documents:
        return False
    
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    chunk_counter = 0
    
    for doc in documents:
        chunks = chunk_text(doc['content'])
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadatas.append({
                'source': doc['filename'],
                'chunk_id': i
            })
            chunk_counter += 1
    
    # Add in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            ids=all_ids[i:batch_end],
            metadatas=all_metadatas[i:batch_end]
        )
    
    return True


def detect_indonesian(text):
    """Detect if query is in Indonesian"""
    indonesian_words = [
        'apa', 'berapa', 'bagaimana', 'mengapa', 'kapan', 'dimana', 'siapa',
        'saya', 'kamu', 'kalian', 'kami', 'yang', 'ini', 'itu', 'dan',
        'untuk', 'dari', 'ke', 'di', 'pada', 'adalah', 'bisa', 'dapat'
    ]
    
    text_lower = text.lower()
    count = sum(1 for word in indonesian_words if word in text_lower)
    return count >= 2


def translate_to_english(query):
    """Translate Indonesian query to English"""
    try:
        response = chat_client.chat.completions.create(
            model='openai/gpt-oss-20b:free',
            messages=[
                {
                    'role': 'system',
                    'content': 'Translate Indonesian to English. Output only the translation.'
                },
                {'role': 'user', 'content': query}
            ],
            timeout=15
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query


def search_documents(query, collection, n_results=3):
    """Search for relevant chunks"""
    try:
        search_query = query
        if detect_indonesian(query):
            search_query = translate_to_english(query)
        
        results = collection.query(
            query_texts=[search_query],
            n_results=n_results
        )
        
        relevant_chunks = []
        for i in range(len(results['documents'][0])):
            relevant_chunks.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i]
            })
        
        return relevant_chunks
    except Exception:
        return []


def generate_answer(query, relevant_chunks, history):
    """Generate answer using LLM"""
    try:
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"[{i}] Source: {chunk['source']}\n{chunk['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        user_prompt = f"""User Question: {query}

Context from PDF:
{context}

Answer the question based on the context above in Bahasa Indonesia."""
        
        history.append({'role': 'user', 'content': user_prompt})
        
        response = chat_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.7,
            max_tokens=600,
            timeout=30
        )
        
        answer = response.choices[0].message.content
        
        history[-1] = {'role': 'user', 'content': query}
        history.append({'role': 'assistant', 'content': answer})
        
        return answer, history
    except Exception:
        return "Maaf, terjadi kesalahan. Silakan coba lagi.", history


def cleanup_collection(collection_name):
    """Delete collection and cleanup"""
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass


def main():
    print("PDF Q&A System")
    print("="*60)
    
    # Step 1: Get PDF path
    pdf_path = input("\nEnter PDF folder path: ").strip()
    
    if not pdf_path:
        print("Invalid path")
        return
    
    # Step 2: Create collection
    collection_name = f"pdf_session_{int(os.times().elapsed * 1000)}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    print("\nLoading documents...")
    
    # Step 3: Load PDFs
    success = load_documents_to_db(pdf_path, collection)
    
    if not success:
        print("No PDF files found or error loading documents")
        cleanup_collection(collection_name)
        return
    
    print("Documents loaded successfully\n")
    
    # Step 4: Q&A Loop
    system_prompt = """You are a helpful AI assistant. Answer questions based on the provided PDF context. Be clear, accurate, and concise. Always answer in Bahasa Indonesia."""
    
    history = [{'role': 'system', 'content': system_prompt}]
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                break
            
            # Search
            results = search_documents(query, collection, n_results=3)
            
            if not results:
                print("AI: Maaf, saya tidak menemukan informasi yang relevan.\n")
                continue
            
            # Generate answer
            answer, history = generate_answer(query, results, history)
            
            print(f"AI: {answer}\n")
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            print(f"AI: Maaf, terjadi kesalahan.\n")
    
    # Step 5: Cleanup
    print("\nCleaning up...")
    cleanup_collection(collection_name)
    print("Session ended\n")


if __name__ == "__main__":
    main()