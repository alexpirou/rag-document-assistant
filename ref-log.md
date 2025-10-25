# Reference Log

## Resources Used

**Documentation:**
- LangChain docs - text splitters and ChromaDB integration
- ChromaDB docs - vector database setup and similarity search
- Streamlit docs - session state and file uploads
- OpenAI API docs - chat completions and embeddings

**Course Materials:**
- `langgraph_chroma_retreiver.ipynb` notebook from class
- `chat_with_pdf.py` starter template
- Lecture notes on RAG systems

**Tools:**
- GitHub Codespaces for development

## GitHub Copilot Usage

Used Copilot extensively as a learning tool throughout the assignment. Instead of asking it to write code, I mostly asked questions to understand concepts and debug issues.

**Main things I learned with Copilot:**

1. **Chunking strategies** - Asked why overlap matters and learned about preventing information loss at boundaries.

2. **Streamlit session state** - Was confused about why data wasn't persisting and learned that Streamlit reruns the whole script on every interaction.

3. **File handling bug** - Was having issues reading PDFs and getting empty content. Copilot helped me trace through the code to see the file pointer was at the end after the first read.

4. **API model naming** - Got a 400 error then learned Cornell's API needs the provider prefix (like "openai.gpt-4o" not just "gpt-4o")

5. **Message flow to LLM** - Responses weren't using document context. Walked through the message array structure to understand when context was being included vs not.

6. **LangChain version conflicts** - Hit a few import errors and received assistance for package compatibility

**How I used it:**

Always asked "why" questions instead of direct code generation questions. When I got stuck, I'd ask it to explain what was happening rather than fix it for me. Made sure I understood each concept before moving forward.

**What I learned:**

The biggest takeaway was understanding the full RAG pipeline and how retrieval actually works, why vector embeddings are useful, and how to structure the context for the LLM.

Using Copilot as a teaching assistant instead of a code generator made this way more valuable as a learning experience.