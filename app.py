import streamlit as st
from crawl4ai import *
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import re
import os
from aiohttp.client_exceptions import ClientConnectorError, InvalidURL

# --- 1. Constants ---
# Session state keys
SESSION_KEYS = {
    "crawled_text": "crawled_text",
    "llmed_text": "llmed_text",
    "summary_text": "summary_text",
    "url_to_crawl": "url_to_crawl"
}

# Gemini prompts
PROMPTS = {
    "extraction": '''
        The given text is the conversion of HTML content into Markdown text.
        Extract the main content while preserving its original wording and substance completely. Your task is:

        1. Maintain the exact language and terminology used in the main content
        2. Remove only clearly irrelevant elements like:
        - Navigation menus
        - Advertisement sections
        - Cookie notices
        - Footers with site information
        - Sidebars with external links
        - Any UI elements that don't contribute to main content
        3. Do not change or rewrite the actual content, but organize it into clear paragraphs for easier reading

        The goal is to create a clean markdown version that reads exactly like the original article, 
        keeping all valuable content but free from distracting elements. Imagine you're creating 
        a perfect reading experience where nothing valuable is lost, but all noise is removed.

        Here is the text to process:

        ''',
    "summary": '''
        Summarize the following markdown content into a concise summary in markdown format, 
        preserving all key points and main ideas. Focus on clarity and brevity, but do not omit important information.

        Content to summarize:
        '''
}

# --- 2. Async Event Loop Setup ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- 3. Core Logic Functions ---
def initialize_session_state():
    """Initializes session state."""
    for key in SESSION_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = ""

def is_valid_url(url):
    """A simple function to validate a URL."""
    regex = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

async def crawl_url(url, config):
    """Asynchronously crawls a URL and saves the result to session_state."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        st.session_state[SESSION_KEYS["crawled_text"]] = result.markdown.raw_markdown
        # Reset LLM text and summary after crawling
        st.session_state[SESSION_KEYS["llmed_text"]] = ""
        st.session_state[SESSION_KEYS["summary_text"]] = ""


@st.cache_resource
def get_gemini_model():
    """Loads and caches the Gemini model."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Please check your .env file.")
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-flash-lite-latest')

def convert_by_gemini(instruction, text):
    """Converts text using Gemini."""
    model = get_gemini_model()
    if not model:
        return None
    try:
        response = model.generate_content(instruction + text)
        return response.text
    except Exception as e:
        st.error(f"An error occurred during Gemini processing: {e}")
        return None

# --- 4. UI Rendering Functions ---
def render_sidebar():
    """Renders the sidebar UI."""
    st.sidebar.title("AI Web Crawler 🌐")

    with st.sidebar.expander("Crawl Settings", expanded=False):
        excluded_tags_default = ["nav", "footer", "aside", "header", "script", "style", "video", "audio", "form", "input", "button"]
        excluded_tags = st.multiselect(
            "Tags to exclude",
            ["nav", "footer", "aside", "header", "script", "style", "video", "audio", "form", "input", "button"],
            default=excluded_tags_default
        )
        
    st.sidebar.text_input(
        "Enter the URL to crawl",
        key=SESSION_KEYS["url_to_crawl"]
    )

    if st.sidebar.button("Crawl", width="stretch"):
        url = st.session_state[SESSION_KEYS["url_to_crawl"]]
        if not url:
            st.sidebar.warning("Please enter a URL.")
            return

        if not is_valid_url(url):
            st.sidebar.error("Please enter a valid URL (e.g., https://example.com)")
            return
        
        with st.spinner(f"Crawling {url}..."):
            try:
                crawler_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    remove_overlay_elements=True,
                    excluded_tags=excluded_tags,
                    markdown_generator=DefaultMarkdownGenerator(
                        options={"ignore_links": False},
                    ),
                )
                asyncio.run(crawl_url(url, crawler_config))
            except (ClientConnectorError, InvalidURL):
                st.error(f"Failed to connect to URL: {url}. Please check the URL and your network connection.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

def render_main_content():
    """Renders the main content area (tabs)."""
    tab1, tab2, tab3, tab4 = st.tabs(["Crawled", "AI Processed", "Markdown", "Summary"])

    crawled_text = st.session_state.get(SESSION_KEYS["crawled_text"])
    llmed_text = st.session_state.get(SESSION_KEYS["llmed_text"])
    summary_text = st.session_state.get(SESSION_KEYS["summary_text"])

    with tab1:
        if crawled_text:
            st.markdown(crawled_text)
        else:
            st.info("No crawled content. Enter a URL and click the 'Crawl' button.")

    with tab2:
        if crawled_text and not llmed_text:
            with st.spinner("Extracting main content with Gemini..."):
                llmed_text = convert_by_gemini(PROMPTS["extraction"], crawled_text)
                st.session_state[SESSION_KEYS["llmed_text"]] = llmed_text
        
        if llmed_text:
            st.markdown(llmed_text)
        else:
            st.info("No processed content. Please run the crawler first.")

    with tab3:
        if llmed_text:
            st.code(llmed_text, language="markdown")
        else:
            st.info("No processed content.")

    with tab4:
        if llmed_text and not summary_text:
            with st.spinner("Generating summary with Gemini..."):
                summary_text = convert_by_gemini(PROMPTS["summary"], llmed_text)
                st.session_state[SESSION_KEYS["summary_text"]] = summary_text

        if summary_text:
            st.markdown(summary_text)
        else:
            st.info("No summarized content.")


# --- 5. Main Application Execution ---
def main():
    """Main application function."""
    st.set_page_config(layout="wide", page_title="AI Web Crawler", page_icon="🌐")
    
    # Load .env file
    load_dotenv()

    # Initialize session state
    initialize_session_state()

    # Render UI
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()