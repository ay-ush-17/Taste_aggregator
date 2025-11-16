import os
import time # <-- Make sure time is imported
import re
import html as _html_module
import pickle
import json
from pathlib import Path
from uuid import uuid4
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
try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        # for type-checkers only; avoid import-time errors in environments without langchain
        from langchain.schema import Document  # type: ignore

    try:
        # prefer the real Document when available at runtime
        from langchain.schema import Document as _LC_Document
        Document = _LC_Document
    except Exception:
        # Lightweight runtime fallback if LangChain isn't installed
        class Document:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}
                # provide a stable id attribute for compatibility with
                # vectorstores or code that expects Document.id
                if 'id' in self.metadata:
                    self.id = self.metadata.get('id')
                else:
                    self.id = str(uuid4())
except Exception:
    # Lightweight fallback if langchain isn't installed in the environment
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
            # ensure an `id` attribute exists for compatibility
            if 'id' in self.metadata:
                self.id = self.metadata.get('id')
            else:
                self.id = str(uuid4())


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
# persistent storage paths
# By default save persistent indexes and metadata to a user-level folder
# outside the project workspace so dev file-watchers (Five/Vite/etc.) don't
# detect frequent changes and trigger reloads.
DEFAULT_PERSIST_DIR = Path.home() / '.taste_aggregator_data'
DATA_DIR = Path(os.environ.get('TASTE_AGG_PERSIST_DIR', DEFAULT_PERSIST_DIR))
FAISS_PICKLE = DATA_DIR / 'faiss_store.pkl'
ARTICLES_JSON = DATA_DIR / 'articles.json'

# folder-based FAISS save (preferred)
FAISS_DIR = DATA_DIR / 'faiss_index'

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


def load_vector_store_if_needed():
    """Try to load persisted FAISS vectorstore into global_vector_store if it's not set."""
    global global_vector_store, embeddings_model
    if global_vector_store is not None:
        return
    # ensure embeddings_model exists for loading
    if embeddings_model is None:
        initialize_models()

    # Try folder-based FAISS load first
    try:
        if FAISS_DIR.exists():
            try:
                print(f"Attempting to load FAISS from folder {FAISS_DIR}")
                global_vector_store = FAISS.load_local(str(FAISS_DIR), embeddings_model)
                print("Loaded FAISS vectorstore from folder.")
                return
            except Exception as e:
                print('FAISS.load_local failed:', e)
    except Exception:
        pass


def _call_llm_and_extract_text(llm_obj, prompt):
    """Call an LLM object using several fallbacks and extract a plain-text response.

    This function attempts common call patterns and handles return shapes from
    different LangChain/third-party wrappers (strings, dicts, objects with
    `.generations`, `.message`, `.content`, or `.text`). It returns a string
    or `None` if no usable text could be extracted.
    """
    if llm_obj is None:
        return None

    def _extract(candidate):
        # normalize common shapes to text
        try:
            if candidate is None:
                return None
            if isinstance(candidate, str):
                return candidate
            # dict-like
            if isinstance(candidate, dict):
                for k in ('text', 'content', 'answer'):
                    if k in candidate and isinstance(candidate[k], str):
                        return candidate[k]
                # sometimes nested
                for v in candidate.values():
                    t = _extract(v)
                    if t:
                        return t
            # objects with attributes
            if hasattr(candidate, 'content') and isinstance(getattr(candidate, 'content'), str):
                return getattr(candidate, 'content')
            if hasattr(candidate, 'text') and isinstance(getattr(candidate, 'text'), str):
                return getattr(candidate, 'text')
            # langchain-like generation objects
            if hasattr(candidate, 'generations'):
                gens = getattr(candidate, 'generations')
                try:
                    # gens is typically a list of lists of Generation
                    first = gens[0][0]
                except Exception:
                    try:
                        first = gens[0]
                    except Exception:
                        first = None
                if first is not None:
                    # try typical attrs
                    for attr in ('text', 'content'):
                        if hasattr(first, attr):
                            val = getattr(first, attr)
                            if isinstance(val, str):
                                return val
                    # sometimes it's a dict-like
                    return _extract(first)
            # chat-style message objects
            if hasattr(candidate, 'message'):
                msg = getattr(candidate, 'message')
                return _extract(msg)
            # list -> try first element
            if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
                return _extract(candidate[0])
        except Exception:
            return None
        return None

    # If this is a Google chat-style model, prefer using .invoke() (modern LangChain interface)
    try:
        if isinstance(llm_obj, ChatGoogleGenerativeAI) or llm_obj.__class__.__name__.lower().find('google') != -1:
            try:
                # Modern LangChain chat models support .invoke(input_string) which returns an AIMessage with .content.
                # This is cleaner and more compatible than .generate() with custom message shims.
                result = llm_obj.invoke(prompt)
                # result should be an AIMessage; extract its .content
                if hasattr(result, 'content') and isinstance(result.content, str):
                    print('LLM returned via invoke(); content length:', len(result.content))
                    return result.content
                # fallback: try extracting generically
                text = _extract(result)
                if text:
                    print('LLM returned via invoke() after extraction; type=', type(result))
                    return text
            except Exception as e:
                print('LLM invoke() branch raised:', e)

    except Exception:
        # defensive: if isinstance check fails for any reason, continue
        pass

    # Try calling in safe ways
    try:
        # If the LLM is a simple callable wrapper that returns text
        if callable(llm_obj):
            resp = llm_obj(prompt)
            text = _extract(resp)
            if text:
                print('LLM call returned via callable branch; type=', type(resp))
                return text
    except Exception as e:
        print('LLM callable branch raised:', e)

    # Try `.generate` if available
    try:
        if hasattr(llm_obj, 'generate'):
            gen = llm_obj.generate([prompt])
            text = _extract(gen)
            if text:
                print('LLM returned via generate(); type=', type(gen))
                return text
    except Exception as e:
        print('LLM generate branch raised:', e)

    # Try `.predict` or `.call` style
    try:
        if hasattr(llm_obj, 'predict'):
            pred = llm_obj.predict(prompt)
            text = _extract(pred)
            if text:
                print('LLM returned via predict(); type=', type(pred))
                return text
    except Exception as e:
        print('LLM predict branch raised:', e)

    try:
        if hasattr(llm_obj, 'call'):
            called = llm_obj.call(prompt)
            text = _extract(called)
            if text:
                print('LLM returned via call(); type=', type(called))
                return text
    except Exception as e:
        print('LLM call branch raised:', e)

    # If everything failed, return None
    return None

    # Fallback: try pickle
    try:
        if FAISS_PICKLE.exists():
            try:
                with open(FAISS_PICKLE, 'rb') as f:
                    global_vector_store = pickle.load(f)
                print(f"Loaded FAISS vectorstore from pickle {FAISS_PICKLE}")
                return
            except Exception as e:
                print('Failed to load FAISS pickle:', e)
    except Exception:
        pass

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

        if not categories:
            return jsonify({"error": "No categories provided."}), 400

        # Calculate articles per feed and max articles based on number of categories
        # Goal: Show 5 articles per selected category
        num_categories = len(categories)
        articles_per_feed = int(data.get('articles_per_feed', 5))  # 5 articles per feed
        max_articles = articles_per_feed * num_categories  # 5 × number of categories
        
        print(f"Building feed for {num_categories} categories: will fetch up to {max_articles} total articles (5 per category)")
        
        # Default to 4 feeds per category unless overridden
        max_feeds = int(data.get('max_feeds', 4))

        # --- Phase 1: Fetch ---
        print(f"Fetching articles for: {categories}")
        articles = fetch_articles_for_categories(categories, articles_per_feed=articles_per_feed, max_articles=max_articles, max_feeds=max_feeds)
        if not articles:
            return jsonify({"error": "No articles found for these categories."}), 404

        print(f"Fetched {len(articles)} articles:")
        for i, art in enumerate(articles):
            text_len = len(art.get('text', ''))
            word_count = len(art.get('text', '').split())
            print(f"  Article {i+1}: {art.get('title')[:60]}... ({text_len} chars, {word_count} words)")
        print("Splitting text...")

        # Build chunked Documents with metadata so we can map retrieved chunks to articles
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = []
        for idx, art in enumerate(articles):
            text = art.get('text') or ''
            if not text:
                continue
            try:
                chunks = text_splitter.split_text(text)
            except Exception:
                # Fallback to create_documents for compatibility
                chunks = [d.page_content for d in text_splitter.create_documents([text])]

            for chunk_idx, c in enumerate(chunks):
                meta = {
                    'title': art.get('title'),
                    'link': art.get('link'),
                    'source': art.get('source'),
                    'published': art.get('published'),
                    'article_idx': idx,
                    'chunk_idx': chunk_idx,
                    'is_numeric': bool(art.get('is_numeric', False)),
                    'category': art.get('category')
                }
                # provide a stable per-chunk id so downstream code or
                # vectorstores that expect document ids can use it
                meta['id'] = f"doc-{idx}-{chunk_idx}"
                documents.append(Document(page_content=c, metadata=meta))

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
        # Persist vector store and article metadata to disk for faster reuse
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Preferred: save FAISS via vectorstore-native save_local (folder)
            try:
                FAISS_DIR.mkdir(parents=True, exist_ok=True)
                # db is a LangChain FAISS wrapper with save_local
                db.save_local(str(FAISS_DIR))
                print(f"Saved FAISS store (folder) to {FAISS_DIR}")
            except Exception as e:
                print('FAISS.save_local failed, falling back to pickle:', e)

            # Fallback: pickle the object for compatibility
            try:
                with open(FAISS_PICKLE, 'wb') as f:
                    pickle.dump(db, f)
                print(f"Persisted FAISS pickle to {FAISS_PICKLE}")
            except Exception as e:
                print('Pickle persistence failed:', e)

            # Save original articles list (with metadata) so we can attach citations later
            with open(ARTICLES_JSON, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False)
            print(f"Persisted articles to {ARTICLES_JSON}")
            # Also log the effective data directory for debugging so you know which
            # path is being written and why file-watchers might have detected changes.
            print(f"Effective DATA_DIR = {DATA_DIR}")
        except Exception as e:
            print(f"Failed to persist vector store or articles: {e}")
        print("Vector store built successfully.")

        # also keep article_texts list for summarization
        article_texts = [a.get('text') for a in articles]

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
                        print(f"  Summarizing: title={art.get('title')[:50]}... input_words={input_words}")
                        
                        # Skip summarizer if input is too short; just use the text as-is
                        if input_words < 50:
                            print(f"    Input too short ({input_words} words), using as-is")
                            long_sum = short
                            print(f"    Using full text ({len(long_sum.split())} words): {long_sum[:150]}...")
                        else:
                            target_words = 80  # Increased from 60 for fuller summaries
                            if input_words <= 0:
                                max_out = 48  # Increased from 32
                            elif input_words < target_words:
                                max_out = max(int(input_words * 0.9), 40)  # Use 90% of input, min 40 tokens
                            else:
                                max_out = target_words
                            max_out = min(max_out, 800)  # Increased cap from 500 to 800
                            min_out = max(20, int(max_out * 0.3))  # Relaxed from 0.4 to 0.3
                            
                            print(f"    Summarizing with max_tokens={max_out}, min_tokens={min_out}")

                            # Generate a summary with the chosen lengths
                            long_sum = summarizer(short, max_length=max_out, min_length=min_out, do_sample=False)[0]['summary_text']
                            print(f"    Generated summary ({len(long_sum.split())} words): {long_sum[:150]}...")

                        # Derive a key finding - use first sentence or ~25 words
                        words = long_sum.split()
                        key_len = 25  # Increased from 12 to capture full sentences
                        
                        # Try to get a complete sentence
                        sents = _top_n_sentences(long_sum, 1)
                        if sents and len(sents[0].split()) >= 10:
                            key_finding = sents[0].strip()
                        else:
                            key_finding = ' '.join(words[:key_len]).strip()
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
                        # fallback: take the first 80 words from the chunk
                        fallback_words = chunk.split()[:80]
                        long_sum = ' '.join(fallback_words)
                        key_finding = ' '.join(fallback_words[:25])  # Increased from 15 to 25 words
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
                    long_sum = ' '.join(words[:80])  # Increased from 60 words
                    key_finding = ' '.join(words[:25])  # Increased from 12/15 to 25 words
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

            payload = {
                "summary": final_summary_clean,
                "per_article_summaries": per_article_clean,
                "summary_html": final_summary_html,
                "per_article_html": per_article_html
            }
            try:
                # Log summary payload metadata for debugging (don't print huge content)
                print("Returning summary payload: keys=", list(payload.keys()),
                      "per_article_count=", len(payload.get('per_article_summaries', [])))
            except Exception:
                pass
            return jsonify(payload)
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
    
    # Try to load persisted vector store if needed
    load_vector_store_if_needed()
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
            # Ensure persisted store is loaded
            load_vector_store_if_needed()

            # Build a retriever if available; otherwise use the vectorstore object itself.
            if hasattr(global_vector_store, 'as_retriever'):
                retriever = global_vector_store.as_retriever()
            else:
                retriever = global_vector_store

            # Diagnostic logging to help debug which methods are available
            try:
                available = [m for m in ('get_relevant_documents','retrieve','similarity_search','similarity_search_with_score','similarity_search_by_vector') if hasattr(retriever, m) or hasattr(global_vector_store, m)]
                print("Retriever diagnostic: type=", type(retriever), "available_methods=", available)
            except Exception:
                pass

            docs = None
            # Try LangChain-style retriever methods first
            if hasattr(retriever, 'get_relevant_documents'):
                try:
                    docs = retriever.get_relevant_documents(question)
                except Exception as e:
                    print('get_relevant_documents failed:', e)
            elif hasattr(retriever, 'retrieve'):
                try:
                    docs = retriever.retrieve(question)
                except Exception as e:
                    print('retrieve failed:', e)

            # Fallbacks: try vectorstore similarity search methods
            if docs is None and hasattr(global_vector_store, 'similarity_search'):
                try:
                    docs = global_vector_store.similarity_search(question, k=5)
                except Exception as e:
                    print('similarity_search failed:', e)

            # similarity_search_with_score may return [(doc, score), ...]
            if docs is None and hasattr(global_vector_store, 'similarity_search_with_score'):
                try:
                    scored = global_vector_store.similarity_search_with_score(question, k=5)
                    if isinstance(scored, list) and len(scored) and isinstance(scored[0], tuple):
                        print('Top retrieval scores:', [round(s,4) for (_, s) in scored])
                        docs = [d for (d, s) in scored]
                except Exception as e:
                    print('similarity_search_with_score failed:', e)

            if docs is None and hasattr(global_vector_store, 'similarity_search_by_vector') and embeddings_model is not None:
                try:
                    qvec = embeddings_model.embed_query(question)
                    docs = global_vector_store.similarity_search_by_vector(qvec, k=5)
                except Exception as e:
                    print('similarity_search_by_vector failed:', e)

            if not docs:
                print('No documents retrieved for question:', question)
                return jsonify({"answer": "I couldn't find relevant documents to answer that. Please run 'Build & Analyze' or try a different question.", "citations": [], "used_docs_count": 0, "model": 'none'})

            # Normalize docs: some vectorstores return (doc, score) tuples
            if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], tuple):
                docs = [d for (d, _s) in docs]

            top_texts = [getattr(d, 'page_content', str(d)) for d in docs[:5]]
            context = "\n\n".join(top_texts)

            if llm is not None:
                try:
                    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer succinctly:"
                    answer_text = _call_llm_and_extract_text(llm, prompt)
                    if not answer_text:
                        print('LLM produced no extracted text; returning context instead.')
                        return jsonify({"answer": context})
                    print('Generated answer (truncated):', str(answer_text)[:300])
                    return jsonify({"answer": str(answer_text)})
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


@app.route('/rag-chat', methods=['POST'])
def rag_chat():
    """
    RAG chat endpoint: retrieve top-k documents from FAISS and generate a concise answer with citations.
    Request JSON: { query: str, category: str (optional), top_k: int (default 5), max_context_chars: int }
    """
    global global_vector_store, llm, summarizer
    try:
        initialize_models()
        data = request.get_json()
        query = (data.get('query') or data.get('question') or '').strip()
        if not query:
            return jsonify({'error': 'No query provided.'}), 400

        category = data.get('category')
        top_k = int(data.get('top_k', 5))
        max_context_chars = int(data.get('max_context_chars', 4000))

        # Try to load persisted index if in-memory store is missing
        load_vector_store_if_needed()
        if global_vector_store is None:
            return jsonify({'error': 'No vector store available. Please run Build & Analyze first.'}), 400

        # load articles metadata if available
        articles = []
        if ARTICLES_JSON.exists():
            try:
                with open(ARTICLES_JSON, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
            except Exception as e:
                print(f"Failed to load articles metadata: {e}")

        # Build retriever and retrieve documents
        # Build a retriever if available; otherwise use the vectorstore object itself.
        if hasattr(global_vector_store, 'as_retriever'):
            retriever = global_vector_store.as_retriever(search_kwargs={"k": top_k})
        else:
            retriever = global_vector_store

        # Diagnostic logging to help debug which methods are available
        try:
            available = [m for m in ('get_relevant_documents','retrieve','similarity_search','similarity_search_by_vector') if hasattr(retriever, m) or hasattr(global_vector_store, m)]
            print("RAG retriever diagnostic: type=", type(retriever), "available_methods=", available)
        except Exception:
            pass

        docs = None
        # Try LangChain-style retriever methods first
        if hasattr(retriever, 'get_relevant_documents'):
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception as e:
                print('get_relevant_documents failed:', e)
        elif hasattr(retriever, 'retrieve'):
            try:
                docs = retriever.retrieve(query)
            except Exception as e:
                print('retrieve failed:', e)

        # Fallbacks: try vectorstore similarity search methods
        if docs is None:
            if hasattr(global_vector_store, 'similarity_search'):
                try:
                    docs = global_vector_store.similarity_search(query, k=top_k)
                except Exception as e:
                    print('similarity_search failed:', e)

        if docs is None and hasattr(global_vector_store, 'similarity_search_by_vector') and embeddings_model is not None:
            try:
                qvec = embeddings_model.embed_query(query)
                docs = global_vector_store.similarity_search_by_vector(qvec, k=top_k)
            except Exception as e:
                print('similarity_search_by_vector failed:', e)

        if docs is None:
            # If no retrieval method worked, log diagnostics and return a safe client-facing message
            try:
                attrs = dir(retriever) if retriever is not None else dir(global_vector_store)
                print("RAG retrieval failure diagnostic: available attrs (sample):", [a for a in attrs if not a.startswith('_')][:80])
            except Exception:
                pass
            return jsonify({
                'answer': "I couldn't find relevant documents to answer that. Please run 'Build & Analyze' or try a different query.",
                'citations': [],
                'used_docs_count': 0,
                'model': 'none'
            })

        # Helper: map a doc to an article by checking if article text contains doc text
        def _find_article_for_doc(doc):
            text = getattr(doc, 'page_content', None) or str(doc)
            for a in articles:
                if not a or not a.get('text'):
                    continue
                try:
                    if text.strip() and text.strip()[:50] in a.get('text', ''):
                        return a
                except Exception:
                    continue
            return None

        retrieved = []
        context_parts = []
        used_chars = 0
        for i, d in enumerate(docs[:top_k]):
            snippet = getattr(d, 'page_content', None) or str(d)
            # short snippet
            snippet = re.sub(r'\s+', ' ', snippet).strip()[:800]
            # Prefer document metadata (if available) for accurate citations
            doc_meta = getattr(d, 'metadata', None) or {}
            if doc_meta:
                meta = {
                    'title': doc_meta.get('title'),
                    'link': doc_meta.get('link'),
                    'source': doc_meta.get('source'),
                    'published': doc_meta.get('published'),
                    'snippet': snippet,
                    'score': None
                }
            else:
                art = _find_article_for_doc(d)
                meta = {
                    'title': art.get('title') if art else None,
                    'link': art.get('link') if art else None,
                    'source': art.get('source') if art else None,
                    'published': art.get('published') if art else None,
                    'snippet': snippet,
                    'score': None
                }
            part = f"[{i+1}] {meta['title'] or 'Unknown title'} — {meta['source'] or 'Unknown source'}\n{snippet}\nURL: {meta['link'] or 'N/A'}\n"
            if used_chars + len(part) > max_context_chars:
                break
            context_parts.append(part)
            used_chars += len(part)
            retrieved.append(meta)

        if not context_parts:
            return jsonify({'answer': "I couldn't find relevant documents for that query.", 'citations': [], 'used_docs_count': 0})

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = (
            "You are a concise news assistant. Answer the question using only the numbered snippets below. "
            "Provide a short answer (1-3 sentences) and append citation markers like [1] referring to the snippets. "
            "If the snippets don't contain enough info, say you don't know.\n\n"
            f"Snippets:\n{context}\nQuestion: {query}\nAnswer:"
        )

        # Generate an answer using the LLM if available, otherwise use the summarizer as a fallback
        answer_text = None
        model_used = None
        try:
            if llm is not None:
                # Use robust helper to call the LLM and extract text
                answer_text = _call_llm_and_extract_text(llm, prompt)
                if answer_text:
                    model_used = 'google-genai' if 'GOOGLE_API_KEY' in os.environ else 'local-llm'
            if (llm is None or not answer_text) and summarizer is not None:
                # Use the summarizer to produce a short answer by feeding the prompt
                short = ' '.join(context_parts)[:2000]
                max_l = 128
                res = summarizer(short + '\nQuestion: ' + query, max_length=max_l, min_length=20, do_sample=False)
                answer_text = res[0]['summary_text'] if isinstance(res, list) and res else str(res)
                model_used = 'hf-summarizer'
            else:
                answer_text = "I don't have a model available to generate an answer right now."
                model_used = 'none'
        except Exception as e:
            print(f"RAG generation failed: {e}")
            answer_text = "Failed to generate answer due to an internal error."
            model_used = 'error'

        return jsonify({'answer': str(answer_text), 'citations': retrieved, 'used_docs_count': len(retrieved), 'model': model_used})

    except Exception as e:
        print(f"Error in /rag-chat: {e}")
        return jsonify({'error': str(e)}), 500

# --- Run the App ---
if __name__ == "__main__":
    # We add use_reloader=False because the global_vector_store
    # would be reset every time the file is auto-saved.
    app.run(debug=True, port=5000, use_reloader=False)