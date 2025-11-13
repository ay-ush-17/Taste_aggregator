import os
import time # <-- Make sure time is imported
import re
import html as _html_module
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our new pipeline function
from ml.pipeline import fetch_articles_for_categories

# --- LangChain & GenAI Imports ---
# import google.generativeai as genai  <-- WE REMOVED THIS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- THIS line still used
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddings:
    """A very small adapter to provide embed_documents/embed_query like LangChain expects."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Expect a list of strings
        return [self.model.encode(t) for t in texts]

    def embed_query(self, text):
        return self.model.encode(text)
    
    def __call__(self, texts):
        """Allow the object to be used as a callable embedding function (compat shim)."""
        # Accept either a single string or a list of strings
        if isinstance(texts, str):
            return self.embed_documents([texts])[0]
        return self.embed_documents(texts)

# --- App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for our frontend

# --- API Key Configuration ---
# CRITICAL: You must set this in your terminal BEFORE running the app.
# In your terminal, run:
# $env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  (for PowerShell)
# export GOOGLE_API_KEY="YOUR_API_KEY_HERE" (for Bash/Git Bash)
#
# DO NOT paste your key directly into this file.

# We check if the key exists, but we don't need genai.configure()
# The new LangChain libraries will find the key automatically.
if "GOOGLE_API_KEY" not in os.environ:
    print("="*50)
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set the key and restart the application.")
    print("="*50)

# --- Global Variables ---
# We will store our vector store in memory.
# This means it will be reset every time the server restarts.
global_vector_store = None
llm = None
embeddings_model = None
summarizer = None

def initialize_models():
    """Initialize the LLM and Embedding models to be reused."""
    global llm, embeddings_model, summarizer
    # Initialize summarizer (HF) first; it's independent of Google API
    if summarizer is None:
        try:
            # reduce noisy warnings from the transformers library
            try:
                # local import to avoid global dependency for logging API
                from transformers import logging as _transformers_logging
                _transformers_logging.set_verbosity_error()
            except Exception:
                pass

            summarizer = hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            summarizer = None

    # Use local sentence-transformers for embeddings by default (hybrid approach).
    # Google GenAI will be used only for the LLM when GOOGLE_API_KEY is present.
    if embeddings_model is None:
        embeddings_model = SentenceTransformersEmbeddings()

    if "GOOGLE_API_KEY" in os.environ:
        if llm is None:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0.3, convert_system_message_to_human=True)
        print("Initialized Google GenAI LLM; embeddings will be computed locally.")
    else:
        print("Running in local-only mode: embeddings and summarization will be local unless GOOGLE_API_KEY is set.")

# --- API Endpoints ---

@app.route("/")
def home():
    """A simple health-check endpoint."""
    return "GenAI InsightFeed API is running!"

@app.route("/build-and-summarize", methods=['POST'])
def build_and_summarize():
    """
    Fetches articles, builds a vector store, and generates a summary.
    This is the main "RAG" pipeline.
    """
    global global_vector_store, llm, embeddings_model
    
    try:
        initialize_models() # Make sure models are loaded
        
        data = request.get_json()
        categories = data.get('categories')

        # Default to 2 articles per feed unless overridden
        articles_per_feed = int(data.get('articles_per_feed', 2))
        max_articles = data.get('max_articles')
        max_articles = int(max_articles) if max_articles is not None else None
        # Default to 4 feeds per category unless overridden
        max_feeds = int(data.get('max_feeds', 4))

        if not categories:
            return jsonify({"error": "No categories provided."}), 400

        # --- Phase 1: Fetch ---
        print(f"Fetching articles for: {categories}")
        articles = fetch_articles_for_categories(categories, articles_per_feed=articles_per_feed, max_articles=max_articles, max_feeds=max_feeds)
        if not articles:
            return jsonify({"error": "No articles found for these categories."}), 404

        print(f"Fetched {len(articles)} articles. Splitting text...")

        # Extract texts for splitting/embedding
        article_texts = [a.get('text') for a in articles]

        # --- Phase 2: Split & Embed (Build Vector Store) ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.create_documents(article_texts)
        
        print(f"Created {len(documents)} document chunks. Building FAISS vector store...")
        
        # Build the FAISS vector store using local sentence-transformers embeddings.
        db = None
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]

            if db is None:
                db = FAISS.from_documents(batch_docs, embeddings_model)
            else:
                db.add_documents(batch_docs)

            print(f"Embedded batch {i//batch_size + 1} / {max(1, (len(documents)+batch_size-1)//batch_size)}")

        global_vector_store = db
        
        print("Vector store built successfully.")

        # --- Phase 3: Summarize ---
        # We'll use a simple "stuff" chain for the summary
        # Note: This may fail if there are too many articles (token limit)
        # A more robust method is `map_reduce`, but it's slower.
        print("Generating summary...")
        try:
            # Map-Reduce summarization with line constraints:
            # - Each article/chunk -> 3-line summary (map)
            # - Aggregate the per-chunk summaries and produce a final summary of at most 15 lines (reduce)
            # Summarize per original article (we have structured articles)
            texts = article_texts

            def _split_sentences(text):
                # A simple sentence splitter using punctuation.
                import re
                parts = re.split(r'(?<=[.!?])\s+', text.strip())
                parts = [p.strip() for p in parts if p.strip()]
                return parts

            def _top_n_sentences(text, n):
                sents = _split_sentences(text)
                if len(sents) >= n:
                    return sents[:n]
                # Fallback: split by newline then by length
                lines = text.splitlines()
                if len(lines) >= n:
                    return [l.strip() for l in lines[:n]]
                # As a last resort, chunk the text into n roughly equal parts
                if len(text) < 200:
                    return [text]
                chunk_size = max(1, len(text) // n)
                return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)][:n]

            per_chunk_summaries = []
            if summarizer is not None:
                for art, chunk in zip(articles, texts):
                    try:
                        short = chunk[:4000]
                        # choose sensible output lengths based on input size
                        input_words = len(short.split())
                        target_words = 60
                        if input_words <= 0:
                            max_out = 32
                        elif input_words < target_words:
                            max_out = max(int(input_words * 0.6), 16)
                        else:
                            max_out = target_words
                        max_out = min(max_out, 400)
                        min_out = max(12, int(max_out * 0.4))

                        # Generate a summary with the chosen lengths
                        long_sum = summarizer(short, max_length=max_out, min_length=min_out, do_sample=False)[0]['summary_text']

                        # Derive a short key finding of ~10-15 words
                        words = long_sum.split()
                        key_len = 12
                        key_finding = ' '.join(words[:key_len]).strip()
                        if len(words) < 10:
                            # fallback: use first sentence if too short
                            sents = _top_n_sentences(long_sum, 1)
                            key_finding = sents[0] if sents else long_sum

                        per_chunk_summaries.append({
                            'key_finding': key_finding,
                            'summary': long_sum,
                            'title': art.get('title'),
                            'source': art.get('source'),
                            'link': art.get('link'),
                            'is_numeric': art.get('is_numeric', False)
                        })
                    except Exception as e:
                        print(f"Chunk summarization failed: {e}")
                        # fallback: take the first 60 words from the chunk
                        fallback_words = chunk.split()[:60]
                        long_sum = ' '.join(fallback_words)
                        key_finding = ' '.join(fallback_words[:12])
                        per_chunk_summaries.append({
                            'key_finding': key_finding,
                            'summary': long_sum,
                            'title': art.get('title') if isinstance(art, dict) else None,
                            'source': art.get('source') if isinstance(art, dict) else None,
                            'link': art.get('link') if isinstance(art, dict) else None,
                            'is_numeric': art.get('is_numeric', False) if isinstance(art, dict) else False,
                        })
            else:
                # No summarizer: create simple summaries from the chunk text
                for art, chunk in zip(articles, texts):
                    words = chunk.split()
                    long_sum = ' '.join(words[:60])
                    key_finding = ' '.join(words[:12])
                    per_chunk_summaries.append({
                        'key_finding': key_finding,
                        'summary': long_sum,
                        'title': art.get('title') if isinstance(art, dict) else None,
                        'source': art.get('source') if isinstance(art, dict) else None,
                        'link': art.get('link') if isinstance(art, dict) else None,
                        'is_numeric': art.get('is_numeric', False) if isinstance(art, dict) else False,
                    })

            # Reduce step: combine per-chunk summaries (use the 'summary' field) and summarize
            combined = '\n\n'.join([d.get('summary', '') for d in per_chunk_summaries])

            if summarizer is not None:
                try:
                    # helper: choose output lengths dynamically based on input size
                    def _choose_lengths(text, target_words=60, cap_max=400):
                        words = len(text.split())
                        if words <= 0:
                            return (min(32, cap_max), max(12, int(min(32, cap_max)*0.4)))
                        if words < target_words:
                            # don't ask for more output words than input; scale down
                            max_out = max( int(words * 0.6), 16 )
                        else:
                            max_out = target_words
                        max_out = min(max_out, cap_max)
                        min_out = max(12, int(max_out * 0.4))
                        return (max_out, min_out)

                    # If combined is large, summarize in windows and then combine
                    if len(combined) > 8000:
                        mini_summaries = []
                        start = 0
                        window = 3000
                        while start < len(combined):
                            part = combined[start:start+window]
                            try:
                                max_l, min_l = _choose_lengths(part, target_words=80, cap_max=300)
                                ms = summarizer(part, max_length=max_l, min_length=min_l, do_sample=False)[0]['summary_text']
                                mini_summaries.append(ms)
                            except Exception as e:
                                print(f"Mini summarize failed: {e}")
                            start += window
                        reduced_input = '\n\n'.join(mini_summaries)
                    else:
                        reduced_input = combined

                    max_l, min_l = _choose_lengths(reduced_input, target_words=150, cap_max=600)
                    final_raw = summarizer(reduced_input[:10000], max_length=max_l, min_length=min_l, do_sample=False)[0]['summary_text']
                    final_lines = _top_n_sentences(final_raw, 15)
                    final_summary = '\n'.join(final_lines)
                except Exception as e:
                    print(f"Final summarization failed: {e}")
                    # Fallback: take top 15 lines from combined
                    final_summary = '\n'.join(_top_n_sentences(combined, 15))
            else:
                final_summary = '\n'.join(_top_n_sentences(combined, 15))

            print("Summary generated.")
            # Clean up newlines and excessive whitespace for safer client rendering
            def _clean_text(s):
                if s is None:
                    return ""
                t = str(s).replace('\n', ' ')
                t = re.sub(r'\s+', ' ', t).strip()
                return t

            final_summary_clean = _clean_text(final_summary)
            per_article_clean = []
            for d in per_chunk_summaries:
                per_article_clean.append({
                    'key_finding': _clean_text(d.get('key_finding')),
                    'summary': _clean_text(d.get('summary')),
                    'title': d.get('title') or None,
                    'source': d.get('source') or None,
                    'link': d.get('link') or None,
                    'is_numeric': bool(d.get('is_numeric', False)),
                })

            # Build safe HTML snippet for per-article items (list)
            per_article_html_items = []
            for art in per_article_clean:
                title = art['title'] or art['key_finding'][:120]
                source = art['source'] or 'Unknown'
                per_article_html_items.append(f"<li><strong>{_html_module.escape(title)}</strong> — <em>{_html_module.escape(source)}</em></li>")
            per_article_html = '<ul class="article-summaries list-disc pl-6">' + ''.join(per_article_html_items) + '</ul>'
            final_summary_html = f'<p>{_html_module.escape(final_summary_clean)}</p>'

            return jsonify({
                "summary": final_summary_clean,
                "per_article_summaries": per_article_clean,
                "summary_html": final_summary_html,
                "per_article_html": per_article_html
            })
        except Exception as e:
            print(f"Error generating summary: {e}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"Error in /build-and-summarize: {e}")
        # Add the 429 error check back in
        if "429" in str(e):
             return jsonify({"error": f"Error embedding content: 429 You exceeded your current quota. This is a common free-tier limit. Please wait a minute and try again."}), 429
        return jsonify({"error": str(e)}), 500

@app.route("/ask-my-feed", methods=['POST'])
def ask_my_feed():
    """
    Answers a question using the in-memory RAG vector store.
    """
    global global_vector_store, llm
    
    if global_vector_store is None:
        return jsonify({"error": "Feed has not been built. Please 'Build & Analyze' first."}), 400
        
    try:
        initialize_models() # Ensure models are loaded
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided."}), 400

        print(f"Answering question: {question}")
        
        # --- Phase 4: Retrieve & Answer (simple) ---
        try:
            retriever = global_vector_store.as_retriever()
            # Try common retrieval methods
            if hasattr(retriever, 'get_relevant_documents'):
                docs = retriever.get_relevant_documents(question)
            elif hasattr(retriever, 'retrieve'):
                docs = retriever.retrieve(question)
            else:
                return jsonify({"error": "Underlying retriever does not support retrieval methods."}), 500

            top_texts = [getattr(d, 'page_content', str(d)) for d in docs[:5]]
            context = "\n\n".join(top_texts)

            if llm is not None:
                try:
                    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer succinctly:"
                    if callable(llm):
                        answer = llm(prompt)
                    else:
                        answer = str(prompt)
                    return jsonify({"answer": str(answer)})
                except Exception as e:
                    print(f"LLM generation failed, returning retrieved context instead: {e}")
                    return jsonify({"answer": context})
            else:
                return jsonify({"answer": context})

        except Exception as e:
            print(f"Error in retrieval/answering: {e}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"Error in /ask-my-feed: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == "__main__":
    # We add use_reloader=False because the global_vector_store
    # would be reset every time the file is auto-saved.
    app.run(debug=True, port=5000, use_reloader=False)