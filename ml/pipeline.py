import feedparser
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urlparse
import argparse
import json

# --- Phase 2: Category-Based RSS Feeds ---
# A mapping of our frontend categories to a list of RSS feeds.
CATEGORY_FEEDS = {
    "AI": [
        "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "https://www.brookings.edu/topic/artificial-intelligence/feed/",
        "https://thenewstack.io/feed/"
    ],
    "Web3": [
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://blockworks.co/feed",
        "https://www.theblock.co/rss/latest"
    ],
    "StockMarket": [
        "https://www.zacks.com/rss/stocks.php",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.ft.com/world/rss"
    ],
    "Geopolitics": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.atlanticcouncil.org/feed/",
        "https://feeds.npr.org/1001/rss.xml"
    ],
    "Science": [
        "https://www.nature.com/nature.rss",
        "https://phys.org/rss-feed/",
        "https://www.sciencedaily.com/rss/earth_climates/environmental_news.xml"
    ],
    "Business": [
        "https://www.ft.com/management/rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://techcrunch.com/feed/"
    ]
    # You can add even more categories and feeds here
}


# --- Helper Function (Scraper) ---
# This is the same scraper from before, slightly simplified.
def remove_numeric_blocks(text):
    """
    Remove numeric-heavy/price-list blocks from scraped text and return cleaned text.
    This is extracted so both the scraper and RSS-summary fallback can use it.
    """
    # Split into sentences and keep only those that contain alphabetic characters
    parts = re.split(r'(?<=[.!?])\s+', text)
    kept = []
    for s in parts:
        # proportion of digit characters in the sentence
        digit_count = sum(1 for c in s if c.isdigit())
        if len(s.strip()) == 0:
            continue
        digit_ratio = digit_count / max(1, len(s))
        # keep sentence if it has letters and is not dominated by digits
        if re.search(r'[A-Za-z]', s) and digit_ratio < 0.3:
            kept.append(s.strip())
    if len(kept) >= 2:
        return ' '.join(kept)
    # fallback: remove long runs of numeric tokens instead
    cleaned = re.sub(r'(?:\$?\d+[\d,\.]*\s+){6,}', '[numeric data removed] ', text)
    return cleaned


def scrape_article_text(url):
    """
    Fetches and scrapes the main text content from a given URL.
    Returns the plain text of the article.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None # Failed to fetch

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try multiple strategies to find the main content
        main_content = soup.find('article') or soup.find('main') or soup.find('div', {'class': re.compile(r'content|body|post|entry', re.I)})
        if not main_content:
            main_content = soup

        # Remove script, style, and nav elements
        for script in main_content(['script', 'style', 'nav', 'footer']):
            script.decompose()

        # Try to get paragraphs
        paragraphs = main_content.find_all('p')
        
        # If very few paragraphs, try divs with text
        if len(paragraphs) < 3:
            paragraphs = main_content.find_all(['p', 'div'], limit=50)  # Increased from 20
            paragraphs = [p for p in paragraphs if len(p.get_text().strip()) > 20]
        
        if not paragraphs:
            print(f"  Scraper: no content found for {url}")
            return None

        article_text = ' '.join([p.get_text() for p in paragraphs])
        article_text = re.sub(r'\s+', ' ', article_text).strip()

        cleaned = remove_numeric_blocks(article_text)

        # mark numeric-heavy if cleaning changed the text or if digit ratio is high
        digit_count = sum(1 for c in article_text if c.isdigit())
        digit_ratio = digit_count / max(1, len(article_text))
        is_numeric = (cleaned != article_text) or (digit_ratio > 0.25)

        article_text = cleaned

        if len(article_text) < 200: # Filter out very short snippets (relaxed from 300)
            print(f"  Scraper: scraped text too short after cleaning ({len(article_text)} chars) for {url}")
            return None

        # Return a dict to include metadata about numeric-heavy content
        return { 'text': article_text, 'is_numeric': is_numeric }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# --- Main Function for this Module ---

def fetch_articles_for_categories(categories, articles_per_feed=5, max_articles=None, max_feeds=4, min_chars=300):
    """
    Fetches and scrapes articles for a given list of categories.
    
    Args:
        categories (list): A list of category keys (e.g., ["AI", "Geopolitics"])
        articles_per_feed (int): Max number of articles to grab from each feed.
        max_articles (int|None): Optional cap on total number of articles to return across all feeds.
        
    Returns:
        list: A list of all the scraped article texts.
    """
    print(f"Starting to fetch articles for: {categories}")
    all_article_texts = []
    
    # allow case-insensitive category names by mapping lowercase->canonical
    key_map = {k.lower(): k for k in CATEGORY_FEEDS.keys()}
    for category in categories:
        cat_key = key_map.get(category.lower()) if isinstance(category, str) else None
        if not cat_key:
            print(f"Warning: No feeds found for category '{category}'")
            continue
            
        print(f"\nProcessing category: {category}")
        # Optionally limit number of feeds per category
        feed_list = CATEGORY_FEEDS[category]
        if max_feeds is not None:
            feed_list = feed_list[:max_feeds]

        for feed_url in feed_list:
            # Simple fix for potential typo in Business feed
            if "httpscsv.com" in feed_url:
                print(f"  Skipping invalid feed URL: {feed_url}")
                continue

            print(f"  Fetching feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            if getattr(feed, 'bozo', False):
                try:
                    print(f"    Warning: feed parser reported an issue: {getattr(feed, 'bozo_exception', '')}")
                except Exception:
                    pass

            # Limit the number of articles per feed
            for entry in feed.entries[:articles_per_feed]:
                if not hasattr(entry, 'link') or not hasattr(entry, 'title'):
                    continue
                print(f"    Scraping: {entry.title}")
                scraped = scrape_article_text(entry.link)

                # capture published timestamp from RSS entry if available
                published = None
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = time.strftime('%Y-%m-%dT%H:%M:%SZ', entry.published_parsed)
                    else:
                        published = getattr(entry, 'published', None) or getattr(entry, 'updated', None)
                except Exception:
                    published = getattr(entry, 'published', None) or getattr(entry, 'updated', None)

                if scraped and isinstance(scraped, dict) and scraped.get('text'):
                    title = getattr(entry, 'title', None)
                    link = getattr(entry, 'link', None)
                    source = None
                    try:
                        source = urlparse(link).netloc if link else None
                    except Exception:
                        source = None

                    article_obj = {
                        'title': title,
                        'link': link,
                        'source': source,
                        'text': scraped.get('text'),
                        'is_numeric': bool(scraped.get('is_numeric')),
                        'published': published,
                        'category': cat_key
                    }

                    all_article_texts.append(article_obj)
                    print(f"      ...Success, article added ({len(all_article_texts)} total)")
                else:
                    # Try fallback: use RSS entry summary or content if scraping fails
                    fallback_text = None
                    try:
                        # Combine title + summary for better context on short RSS entries
                        title_text = getattr(entry, 'title', '')
                        summary_text = getattr(entry, 'summary', '')
                        fallback_text = (title_text + ' ' + summary_text).strip() if (title_text or summary_text) else None
                        
                        if not fallback_text and 'content' in entry:
                            # entry.content can be a list of dicts
                            content = entry.get('content')
                            if isinstance(content, list):
                                fallback_text = ' '.join([c.get('value','') for c in content if isinstance(c, dict)])
                    except Exception:
                        fallback_text = None

                    if fallback_text:
                        cleaned = remove_numeric_blocks(re.sub(r'\s+', ' ', fallback_text).strip())
                        if len(cleaned) >= 100:  # Very relaxed threshold for RSS fallback
                            title = getattr(entry, 'title', None)
                            link = getattr(entry, 'link', None)
                            source = None
                            try:
                                source = urlparse(link).netloc if link else None
                            except Exception:
                                source = None

                            article_obj = {
                                'title': title,
                                'link': link,
                                'source': source,
                                'text': cleaned,
                                'is_numeric': False,
                                'published': published,
                                'category': cat_key,
                                'fallback_from_rss': True
                            }
                            all_article_texts.append(article_obj)
                            print(f"      ...Fallback summary used, article added ({len(all_article_texts)} total)")
                        else:
                            print(f"      ...Fallback summary too short or numeric for {entry.title}")

                # If we've reached the requested max_articles cap, stop early
                if max_articles is not None and len(all_article_texts) >= max_articles:
                    print(f"Reached max_articles={max_articles}, stopping fetch.")
                    break

                # Be polite
                time.sleep(0.2)

            # If cap reached, break out of feed loop and category loop
            if max_articles is not None and len(all_article_texts) >= max_articles:
                break

        # If cap reached, break out of category loop
        if max_articles is not None and len(all_article_texts) >= max_articles:
            break

    print(f"\nFinished fetching. Total articles scraped: {len(all_article_texts)}")
    return all_article_texts

def _cli_main():
    parser = argparse.ArgumentParser(description='Fetch articles for categories (debug runner)')
    parser.add_argument('--categories', '-c', nargs='+', required=True, help='Category keys (e.g., AI Business)')
    parser.add_argument('--articles-per-feed', '-a', type=int, default=5)
    parser.add_argument('--max-articles', '-m', type=int, default=None)
    parser.add_argument('--min-chars', type=int, default=300)
    parser.add_argument('--out', '-o', type=str, default=None, help='Optional JSON file to write results to')

    args = parser.parse_args()
    results = fetch_articles_for_categories(args.categories, articles_per_feed=args.articles_per_feed, max_articles=args.max_articles, min_chars=args.min_chars)
    print(f"Fetched {len(results)} articles")
    for i, art in enumerate(results[:10], start=1):
        print(f"{i}. {art.get('title')} ({art.get('source')}) - {art.get('link')}")

    if args.out:
        try:
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Wrote {len(results)} articles to {args.out}")
        except Exception as e:
            print(f"Failed to write output file: {e}")


if __name__ == '__main__':
    _cli_main()

# Note: This file can be imported by `app.py` (use fetch_articles_for_categories(...)) or run
# directly for debugging: `python -m ml.pipeline -c AI Business -a 3 -o samples.json`

