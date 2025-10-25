
# Document Q&A Chat App

Upload PDFs or text files and ask questions about them. The app uses RAG (Retrieval-Augmented Generation) to find relevant parts of your documents and generate accurate answers. Built with Streamlit, LangChain, ChromaDB, and OpenAI's API.

## What It Does

- Upload multiple files at once (PDFs, .txt, .md)
- Ask questions about your documents in a chat interface
- Get answers based on actual content from your files
- See exactly which chunks from your documents were used for each answer
- Keep a conversation going with full chat history
- Browse through all the document chunks in the sidebar
- Track token usage so you know what you're spending

## How to Run

Make sure you're in the Codespace environment, then:

1. **Set your API key** (if not already done):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Install everything**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the app**:
   ```bash
   streamlit run chat_with_pdf.py
   ```

4. A popup should appear in the bottom-right - click "Open in Browser". If you miss it, just stop the app (Ctrl+C) and run it again.

### Using the App

1. Click "Browse files" and upload some documents
2. Wait for it to process (you'll see a status indicator)
3. Type your question in the chat box at the bottom
4. The AI answers based on your documents - expand "Source Chunks Used" to see what it referenced

## How It Works

**Processing your documents:**
1. Extracts text (PyPDFLoader for PDFs, direct read for .txt/.md files)
2. Splits into 1000-character chunks with 100-character overlap
3. Converts chunks to embeddings using OpenAI's text-embedding-3-small
4. Stores in ChromaDB (in-memory vector database)

**Answering questions:**
1. Takes your question and converts it to an embedding
2. Finds the 3 most similar chunks from your documents
3. Sends those chunks + your question to GPT-4o-mini
4. Returns the answer (you can see which chunks it used)

## Technical Details

### Chunking Strategy
Using `RecursiveCharacterTextSplitter` with 1000-character chunks and 100-character overlap. This size seemed like a good balance where it includes enough context for the model without hitting token limits. The overlap helps prevent losing information at chunk boundaries.

### Models Used
- **Embeddings**: `openai.text-embedding-3-small` - cheaper and works fine for this
- **LLM**: `openai.gpt-4o-mini` - cheaper than gpt-4o and handles RAG for this use case well
- **Vector DB**: ChromaDB (ephemeral, in-memory)
- **Retrieval**: k=3 chunks which gives enough context without overwhelming the prompt. Tried k=5 but k=3 was better.


## Configuration Changes from Template

### What I Added to requirements.txt

- `langchain-chroma` - for the vector database
- `pypdf` - for reading PDFs (fixed typo from template that had "litellmpypdf")
- `langchain-community` - includes PyPDFLoader
- `langchain-openai` - OpenAI embeddings and LLM integration
- `tiktoken` - for counting tokens

### Version Tweaks

Pinned a few LangChain versions to avoid compatibility issues:
- `langchain-core~=0.2.15`
- `langchain-openai~=0.1.23`
- `langchain-community~=0.2.15`

### API Setup

- Uses `OPENAI_API_KEY` environment variable
- Points to Cornell's API: `https://api.ai.it.cornell.edu`
- Model names need the provider prefix such as `openai.gpt-4o-mini`

## Design Choices

**Chunking:** 1000 characters seemed like a good balance which includes enough context for the model without hitting token limits. The 100-character overlap helps prevent losing information at chunk boundaries. Used RecursiveCharacterTextSplitter because it tries to split on natural boundaries like paragraphs instead of cutting sentences in half.

**Models:** Went with the cheaper options since they work fine for this:
- `text-embedding-3-small` for embeddings
- `gpt-4o-mini` for chat (way cheaper than gpt-4o)

**Retrieval:** k=3 chunks gives enough context without sending too much. Tried k=5 but it didn't really help.

**Storage:** Using ChromaDB in ephemeral mode (in-memory). It's fast and simple for a prototype. Downside is you lose everything when you restart the app, but that's fine for this use case.

**Session State:** Streamlit reruns the whole script on every interaction, so I use session_state to keep the vector database, chunks, and chat history around.

## Troubleshooting

**"No documents processed" error:**
- Make sure you actually uploaded a file
- Check it's a supported format (.txt, .md, or .pdf)
- Try re-uploading

**PDF not working:**
- PDFs need to have actual text, not just scanned images (no OCR)
- Some PDFs with weird formatting might not extract properly
- If it's giving you trouble, try converting to .txt

**Getting API errors (400/401):**
- Check that `OPENAI_API_KEY` is set in your environment
- Make sure you're using Cornell's endpoint
- Model names need the provider prefix: `openai.gpt-4o-mini` not just `gpt-4o-mini`

## Known Issues

- Vector database clears when you restart the app
- Each browser tab has its own separate document collection