import streamlit as st
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import chromadb
import tiktoken

# Clear ChromaDB system cache at the start to initialize script
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

# Initialize session state variables

# Initialize session state variables before anything uses them
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about your uploaded document(s)"}]

if "current_files" not in st.session_state:
    st.session_state["current_files"] = []

# Initialize vectorstore to None if not present
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Initialize token usage tracking
if "token_usage" not in st.session_state:
    st.session_state["token_usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

# Frontend customization

# Page title and header
st.title("Document Q&A")

# Add instructions before uploader
st.markdown("### Upload Your Documents")
st.info("**Tip:** You can upload multiple files at once to ask questions about all of them")

# File uploader widget
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=("txt", "md", "pdf"),
    accept_multiple_files=True,
    help="Supported formats: PDF, TXT, MD"
)

# Show file count
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

# Chat input widget
question = st.chat_input(
    "Ask something about your documents",
    disabled=not uploaded_files,
)

# Allow user to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Ask something about your uploaded document(s)"}]
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Backend File Processing

if uploaded_files:
    # Get list of uploaded filenames
    uploaded_filenames = [f.name for f in uploaded_files]
    
    # Detect if files have changed
    if st.session_state["current_files"] != uploaded_filenames:
        with st.status("Processing documents...", expanded=True) as status:
            st.write("Extracting text...")
    
            st.session_state["current_files"] = uploaded_filenames
    
            # Process all uploaded files
            all_chunks = []
            
            for uploaded_file in uploaded_files:
                # Check file type and extract content accordingly
                if uploaded_file.type == "application/pdf":
                    # For PDFs: save to temp file and use PyPDFLoader
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(uploaded_file.getvalue())
                        temp_pdf_path = temp_pdf.name
                    pdf_loader = PyPDFLoader(temp_pdf_path)
                    pages = pdf_loader.load()
                    file_content = "\n".join([page.page_content for page in pages])
                else:
                    # For .txt and .md: read directly as text
                    file_content = uploaded_file.read().decode("utf-8")
                
                # Chunk the file content
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                )
                file_chunks = text_splitter.split_text(file_content)
                
                # Add chunks from this file to the collection
                all_chunks.extend(file_chunks)

            # Store all chunks in session_state
            st.session_state["file_chunks"] = all_chunks

            st.write("Chunking documents...")
    
            # Create OpenAIEmbeddings with my API config
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                base_url="https://api.ai.it.cornell.edu/v1",
                model="openai.text-embedding-3-small"
            )

            st.write("Creating embeddings...")
    
            # Create Chroma vectorstore from all chunks
            chroma_client = chromadb.EphemeralClient(
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            vectorstore = Chroma.from_texts(
                texts=all_chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name="uploaded_docs"
            )

            # Store vectorstore in session_state
            st.session_state["vectorstore"] = vectorstore

        status.update(label="✅ Documents processed", state="complete")

if question:  # If user has asked a question
    # Get vectorstore from session_state
    vectorstore = st.session_state.get("vectorstore")
    
    # Check if vectorstore exists before querying
    if vectorstore is None:
        st.error("No documents processed. Please upload documents first.")
        st.stop()
    
    try:
        # Use similarity_search to get relevant chunks
        relevant_docs = vectorstore.similarity_search(question, k=3)
        # Format chunks for content sent to llm
        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            # Show context chunks used
            with st.expander("Source Chunks Used", expanded=False):
                for i, doc in enumerate(relevant_docs, 1):
                    st.text_area(
                        f"Content {i}", 
                        doc.page_content, 
                        height=150, 
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    if i < len(relevant_docs):
                        st.divider()
            stream = client.chat.completions.create(
                model="openai.gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Here's relevant context from the uploaded documents:\n\n{context}"},
                    *st.session_state.messages,
                    {"role": "user", "content": question}  # Add question HERE with context present
                ],
                stream=True
            )
            response = st.write_stream(stream)

        # Append both messages after getting response
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.info("Please try re-uploading your documents.")
        # Reset vectorstore to force re-processing
        st.session_state["vectorstore"] = None

# Sidebar rendering

with st.sidebar:
    st.header("About")
    st.markdown("""
    Upload documents and ask questions about their content.
    
    **Supported formats:**
    - PDF (.pdf)
    - Text (.txt)
    - Markdown (.md)
    """)
    if uploaded_files:
        st.divider()
        st.subheader("Uploaded Files")
        for f in uploaded_files:
            st.text(f"• {f.name}")
    
    # Show document chunks in sidebar
    if "file_chunks" in st.session_state and st.session_state["file_chunks"]:
        st.subheader("Advanced Information")

        with st.expander("View Document Chunks"):
            chunks = st.session_state['file_chunks']
            st.write(f"Total chunks: {len(chunks)}")
            
            # Add search bar above chunks
            chunk_search = st.text_input(
                "🔍 Search chunks",
                placeholder="Enter keyword to filter chunks...",
                key="chunk_search"
            )
            
            # Filter chunks based on search query
            if chunk_search:
                filtered_chunks = [
                    (i, chunk) for i, chunk in enumerate(chunks, start=1)
                    if chunk_search.lower() in chunk.lower()
                ]
                if not filtered_chunks:
                    st.info("No chunks match your search query.")
                else:
                    st.write(f"Found {len(filtered_chunks)} matching chunk(s)")
                    for chunk_num, chunk in filtered_chunks:
                        # Highlight search term in the chunk
                        st.text_area(f"Chunk {chunk_num}", chunk, height=100, disabled=True)
            else:
                show_all_chunks = st.checkbox("Show all chunks", value=False, key="show_all_chunks")
                display_chunks = chunks if show_all_chunks else chunks[:5]

                for i, chunk in enumerate(display_chunks, start=1):
                    st.text_area(f"Chunk {i}", chunk, height=100, disabled=True)

                if not show_all_chunks and len(chunks) > 5:
                    st.caption(f"Toggle 'Show all chunks' to view {len(chunks) - 5} more chunks")
        
        # Token usage estimation
        with st.expander("Token Usage Metrics"):
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            
            # Calculate tokens from chat messages
            chat_tokens = 0
            for msg in st.session_state.messages:
                message_content = msg["content"]
                encoded_tokens = encoding.encode(message_content)
                token_count = len(encoded_tokens)
                chat_tokens += token_count
            
            # Estimate embedding tokens (chunks processed)
            chunk_tokens = 0
            for chunk in chunks:
                encoded_chunk = encoding.encode(chunk)
                chunk_token_count = len(encoded_chunk)
                chunk_tokens += chunk_token_count
            
            # Display metrics
            st.metric("Chat Tokens", f"{chat_tokens:,}")
            st.metric("Document Tokens", f"{chunk_tokens:,}")
            st.metric("Total Estimated", f"{chat_tokens + chunk_tokens:,}")
