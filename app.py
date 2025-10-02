import streamlit as st
from crawl4ai import *
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import re
import os
import sys
from aiohttp.client_exceptions import ClientConnectorError, InvalidURL
from weasyprint import HTML
import markdown

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
        Generate a concise summary of the following content.
        The summary must be presented exclusively in Markdown format, utilizing appropriate elements (e.g., bullet points, bolding for emphasis) to ensure high clarity and brevity.
        Crucially, the summary must meticulously preserve all key points and main ideas without omission of essential information. Strive for maximum density of information transfer.

        Content to summarize:
        '''
}

# --- 2. Async Event Loop Setup ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    # Windows 환경에서 ProactorEventLoop를 사용하도록 설정
    if os.name == 'nt': # nt는 Windows를 의미
        try:
            # Proactor 루프가 없는 경우를 대비
            loop = asyncio.ProactorEventLoop()
        except NotImplementedError:
            # Proactor 루프 생성이 불가능하다면 기본 루프 사용
            loop = asyncio.new_event_loop()
    else:
        # Linux/macOS 등은 Selector/Default 루프 사용
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
        if result and hasattr(result, "markdown") and hasattr(result.markdown, "raw_markdown"):
            st.session_state[SESSION_KEYS["crawled_text"]] = result.markdown.raw_markdown
        else:
            st.error("Failed to retrieve markdown content from the crawl result.")
            st.session_state[SESSION_KEYS["crawled_text"]] = ""
        # Reset LLM text and summary after crawling
        st.session_state[SESSION_KEYS["llmed_text"]] = ""
        st.session_state[SESSION_KEYS["summary_text"]] = ""

        if "chat_session" in st.session_state:
            del st.session_state["chat_session"]
        if "chat_display_history" in st.session_state:
            st.session_state["chat_display_history"] = []

def get_gemini_key():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Please check your .env file.")
        return None
    return GEMINI_API_KEY

@st.cache_resource
def get_gemini_model():
    GEMINI_API_KEY = get_gemini_key()
    if not GEMINI_API_KEY:
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-2.5-flash-lite')

@st.cache_data
def convert_by_gemini(instruction, text):
    """Converts text using Gemini."""
    model = get_gemini_model()
    if not model:
        return None
    try:
        response = model.generate_content(instruction + text)
        if not response or not hasattr(response, "text"):
            st.error("Gemini returned no valid response.")
            return None
        return response.text
    except Exception as e:
        st.error(f"An error occurred during Gemini processing: {e}")
        return None

@st.cache_resource
def get_gemini_chat(context):
    model = get_gemini_model()
    if not model:
        return None

    chat = model.start_chat(history=[])
    # Initialize with transcript context
    chat.send_message(
        f"You are an assistant that answers questions about this transcript:\n{context}"
    )
    return chat

def chat_with_gemini(context):
    """Chat interface to ask questions about the content using Gemini API."""
    user_input = st.chat_input("Ask something about the content...")

    model = get_gemini_model()
    if not model:
        return None
    try:
        if "chat_session" not in st.session_state:
            st.session_state["chat_session"] = get_gemini_chat(context)
    except Exception as e:
        st.error(f"Failed to initialize Gemini chat: {e}")
        return None

    if "chat_display_history" not in st.session_state:
        st.session_state["chat_display_history"] = []

    if user_input:
        st.session_state["chat_display_history"].append({"role": "user", "content": user_input})
        try:
            response = st.session_state["chat_session"].send_message(user_input)
            answer = response.text.strip()
            st.session_state["chat_display_history"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Gemini Q&A Error: {e}")

    # Display previous messages in reverse order
    for message in reversed(st.session_state["chat_display_history"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def download_pdf(markdown_text, file_name="crawled_content.pdf"):
    """Converts Markdown to PDF and provides a download button."""
    if not markdown_text:
        st.warning("No content available to generate PDF.")
        return
    # Convert markdown to HTML
    html_text = markdown.markdown(
        markdown_text,
        extensions=['extra', 'codehilite', 'tables', 'fenced_code']
    )
    
    # Read the HTML template from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "styled.html")
    with open(file_path, "r", encoding="utf-8") as f:
        styled_html_template = f.read()
        
    # Replace the placeholder with the actual content
    styled_html = styled_html_template.replace("{{content}}", html_text)
    
    pdf_file = HTML(string=styled_html).write_pdf()
    st.download_button(
        label="Download PDF",
        data=pdf_file,
        file_name=file_name,
        mime="application/pdf"
    )

@st.dialog(title="Markdown Code", width="large")
def show_markdown_code(markdown_text):
    st.code(markdown_text, language="markdown")

# --- 4. UI Rendering Functions ---
def render_sidebar():
    """Renders the sidebar UI."""
    st.sidebar.title("AI Web Crawler 🌐")

    with st.sidebar.expander("Crawl Settings", expanded=False):
        image_handling = st.radio(
            "Image handling",
            ["include", "exclude"],
            index=1,
        )
        image_handling = (image_handling == "exclude")
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
                    exclude_all_images=image_handling,
                    markdown_generator=DefaultMarkdownGenerator(
                        options={"ignore_links": False},
                    ),
                )
                loop.run_until_complete(crawl_url(url, crawler_config))
            except (ClientConnectorError, InvalidURL):
                st.error(f"Failed to connect to URL: {url}. Please check the URL and your network connection.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    
    with st.sidebar.popover("Markdown Code", width="stretch"):
        selected_text = st.radio(
            "Select content to view Markdown code",
            ("Crawled", "AI Processed", "Summary"),
            index=None
        )
        if st.button("Show Markdown", key="show_markdown_button"):
            if selected_text == "Crawled":
                text = st.session_state.get(SESSION_KEYS["crawled_text"])
            elif selected_text == "AI Processed":
                text = st.session_state.get(SESSION_KEYS["llmed_text"])
            else:
                text = st.session_state.get(SESSION_KEYS["summary_text"])

            if text:
                show_markdown_code(text)

    with st.sidebar.popover("Download PDF", width="stretch"):
        selected_text = st.radio(
            "Select content to download as PDF",
            ("Crawled", "AI Processed", "Summary"),
            index=None
        )
        if selected_text == "Crawled":
            text = st.session_state.get(SESSION_KEYS["crawled_text"])
            file_name = "crawled.pdf"
        elif selected_text == "AI Processed":
            text = st.session_state.get(SESSION_KEYS["llmed_text"])
            file_name = "ai_processed.pdf"
        else:
            text = st.session_state.get(SESSION_KEYS["summary_text"])
            file_name = "summary.pdf"

        if text:
            download_pdf(text, file_name=file_name)
    
def render_main_content():
    """Renders the main content area (tabs)."""
    tab1, tab2, tab3, tab4 = st.tabs(["Crawled", "AI Processed", "Summary", "Chatbot"])

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
        if llmed_text and not summary_text:
            with st.spinner("Generating summary with Gemini..."):
                summary_text = convert_by_gemini(PROMPTS["summary"], llmed_text)
                st.session_state[SESSION_KEYS["summary_text"]] = summary_text

        if summary_text:
            st.markdown(summary_text)
        else:
            st.info("No summarized content.")
    
    with tab4:
        if crawled_text:
            chat_with_gemini(crawled_text)
        else:
            st.info("No crawled content to chat about. Please run the crawler first.")

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