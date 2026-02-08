import asyncio
import os
import re
import sys

import markdown
import streamlit as st
from aiohttp.client_exceptions import ClientConnectorError, InvalidURL
from crawl4ai import *
from dotenv import load_dotenv
from google import genai
from weasyprint import HTML
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from utils import fix_markdown_symbol_issue

# --- 1. Constants ---
MODEL_OPTIONS = [
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
    "gemini-3-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
# Session state keys
SESSION_KEYS = {
    "crawled_text": "crawled_text",
    "llmed_text": "llmed_text",
    "translated_text": "translated_text",
    "summary_text": "summary_text",
    "url_to_crawl": "url_to_crawl",
    "selected_model": "selected_model",  # Added for model selection
}

# Gemini prompts
PROMPTS = {
    "extraction": """
    The given text is the conversion of HTML content into Markdown text.
    Extract the main content while preserving its original wording and substance completely. Your task is:

        1. Maintain the exact language and terminology used in the main content
        2. Remove only clearly irrelevant elements like:
        - Navigation menus
        - Advertisement sections
        - Cookie notices, consent banners, and policy links
        - Privacy policy notices, terms of service, and legal disclaimers
        - Popups, modal dialogs, and newsletter signup forms
        - Footers with site information
        - Sidebars with external links
        - Any UI elements that don't contribute to main content
        3. Do not change or rewrite the actual content, but organize it into clear paragraphs for easier reading

        The goal is to create a clean markdown version that reads exactly like the original article,
        keeping all valuable content but free from distracting elements. Imagine you're creating
        a perfect reading experience where nothing valuable is lost, but all noise is removed.

        Here is the text to process:
        """,
    "summary": """
        Analyze the following text and provide a piece of organized summary with the following structure:

        1.  **Title:** (A concise and descriptive title for the text)
        2.  **Executive Summary:** (2-3 sentences providing a high-level overview of the text's purpose and key takeaways)
        3.  **Detailed Breakdown:**  Organize the text into coherent paragraphs, elaborating on the key points. Each paragraph must have a subtitle. Remove any filler words, greetings, or repetitive phrases that do not contribute to a clear understanding of the text's core message.

        Here is the text to summarize:
        """,
    "translation": """
        Translate the following text into Korean.
        Ensure the translation is natural and accurate, preserving the original meaning and tone.
        The output must be in Markdown format.

        Here is the text to translate:
        """,
}

# --- 2. Async Event Loop Setup ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    # Set to use ProactorEventLoop in Windows environment
    if os.name == "nt":  # nt means Windows
        try:
            # In case Proactor loop is not available
            loop = asyncio.ProactorEventLoop()
        except NotImplementedError:
            # If Proactor loop creation is impossible, use default loop
            loop = asyncio.new_event_loop()
    else:
        # Use Selector/Default loop for Linux/macOS, etc.
        loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)


# --- 3. Core Logic Functions ---
def initialize_session_state():
    """Initializes session state."""
    for key in SESSION_KEYS.values():
        if key not in st.session_state:
            if key == SESSION_KEYS["selected_model"]:
                st.session_state[key] = MODEL_OPTIONS[0]
            else:
                st.session_state[key] = ""

    if "chat_display_history" not in st.session_state:
        st.session_state["chat_display_history"] = []


def is_valid_url(url):
    """A simple function to validate a URL."""
    regex = re.compile(
        r"^(?:http|ftp)s?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, url) is not None


def get_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_youtube_transcript(video_id):
    """Fetches the transcript of a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try to fetch English transcript first
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            # Fallback: take the first available transcript
            transcript = next(iter(transcript_list))

        return " ".join([item.text for item in transcript.fetch()])

    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None


async def crawl_url(url, config):
    """Asynchronously crawls a URL and saves the result to session_state."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        if (
            result
            and hasattr(result, "markdown")
            and hasattr(result.markdown, "raw_markdown")
        ):
            st.session_state[SESSION_KEYS["crawled_text"]] = (
                result.markdown.raw_markdown
            )
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
def get_gemini_client():
    GEMINI_API_KEY = get_gemini_key()
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def get_response_text(response):
    """
    Extracts text from Gemini response, filtering out non-text parts to avoid warnings.
    """
    if not response:
        return ""

    # Check if we have candidates and parts to manually extract text
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            if text_parts:
                return "".join(text_parts)

    # Fallback to .text if manual extraction fails
    if hasattr(response, "text"):
        return response.text

    return ""


@st.cache_data
def convert_by_gemini(instruction, text):
    """Converts text using Gemini."""
    client = get_gemini_client()
    if not client:
        return None

    selected_model = st.session_state.get(
        SESSION_KEYS["selected_model"], MODEL_OPTIONS[0]
    )

    try:
        response = client.models.generate_content(
            model=selected_model, contents=instruction + text
        )
        if not response:
            st.error("Gemini returned no valid response.")
            return None

        text = get_response_text(response)
        if not text:
            st.error("Gemini returned no text in response.")
            return None

        return fix_markdown_symbol_issue(text.strip())
    except Exception as e:
        st.error(f"An error occurred during Gemini processing: {e}")
        return None


def initialize_chat_session(context):
    """
    Initialize or reset chat session with transcript context.
    Args:
        context (str): The transcript text to provide context for the chat
    """
    try:
        # Create new chat session with Gemini
        st.session_state["chat_session"] = get_gemini_chat(context)
        st.session_state["chat_display_history"] = []
    except Exception as e:
        st.error(f"Failed to initialize Gemini chat: {e}")
        # Reset session state on error
        st.session_state["chat_session"] = None
        st.session_state["chat_display_history"] = []


def get_gemini_chat(context):
    client = get_gemini_client()
    if not client:
        return None

    selected_model = st.session_state.get(
        SESSION_KEYS["selected_model"], MODEL_OPTIONS[0]
    )

    chat = client.chats.create(model=selected_model, history=[])
    # Initialize with transcript context
    chat.send_message(
        f"You are an assistant that answers questions about this transcript:\n{context}"
    )
    return chat


def chat_with_gemini(context):
    """
    Implements chat interface for transcript Q&A using Gemini API.
    Args:
        context (str): The transcript text to chat about
    """
    # Initialize chat session if it doesn't exist
    if (
        "chat_session" not in st.session_state
        or st.session_state["chat_session"] is None
    ):
        initialize_chat_session(context)

    # Create chat input interface
    user_input = st.chat_input("Ask something about the video...")

    # Process user message and get AI response
    if user_input:
        # Add user message to chat history
        st.session_state["chat_display_history"].append(
            {"role": "user", "content": user_input}
        )
        try:
            # Get AI response
            response = st.session_state["chat_session"].send_message(user_input)
            answer = fix_markdown_symbol_issue(get_response_text(response).strip())
            # Add AI response to chat history
            st.session_state["chat_display_history"].append(
                {"role": "assistant", "content": answer}
            )
        except Exception as e:
            st.error(f"Gemini Q&A Error: {e}")
            # Reset chat session on error
            initialize_chat_session(context)
            return

    # Display chat messages
    for message in st.session_state["chat_display_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def download_pdf(markdown_text):
    """Converts Markdown to PDF and returns the PDF data."""
    if not markdown_text:
        return None

    # Convert markdown to HTML
    html_text = markdown.markdown(
        markdown_text, extensions=["extra", "codehilite", "tables", "fenced_code"]
    )

    # Read the HTML template from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "styled.html")
    with open(file_path, "r", encoding="utf-8") as f:
        styled_html_template = f.read()

    # Replace the placeholder with the actual content
    styled_html = styled_html_template.replace("{{content}}", html_text)

    return HTML(string=styled_html).write_pdf()


@st.dialog(title="Markdown Code", width="large")
def show_markdown_code(markdown_text):
    st.code(markdown_text, language="markdown")


# --- 4. UI Rendering Functions ---
def render_sidebar():
    """Renders the sidebar UI."""
    st.sidebar.title("AI Web Crawler 🌐")

    st.sidebar.selectbox(
        "Select Gemini Model",
        MODEL_OPTIONS,
        key=SESSION_KEYS["selected_model"],
    )

    with st.sidebar.expander("Crawl Settings", expanded=False):
        image_handling = st.radio(
            "Image handling",
            ["include", "exclude"],
            index=1,
        )
        image_handling = image_handling == "exclude"
        excluded_tags_default = [
            "nav",
            "footer",
            "aside",
            "header",
            "script",
            "style",
            "video",
            "audio",
            "form",
            "input",
            "button",
        ]
        excluded_tags = st.multiselect(
            "Tags to exclude",
            [
                "nav",
                "footer",
                "aside",
                "header",
                "script",
                "style",
                "video",
                "audio",
                "form",
                "input",
                "button",
            ],
            default=excluded_tags_default,
        )

    st.sidebar.text_input("Enter the URL to crawl", key=SESSION_KEYS["url_to_crawl"])

    if st.sidebar.button("Crawl", width="stretch"):
        url = st.session_state[SESSION_KEYS["url_to_crawl"]]
        if not url:
            st.sidebar.warning("Please enter a URL.")
            return

        if not is_valid_url(url):
            st.sidebar.error("Please enter a valid URL (e.g., https://example.com)")
            return

        video_id = get_video_id(url)
        if video_id:
            with st.spinner(f"Fetching transcript for {url}..."):
                transcript = get_youtube_transcript(video_id)
                if transcript:
                    st.session_state[SESSION_KEYS["crawled_text"]] = transcript
                    # Reset LLM text and summary after crawling
                    st.session_state[SESSION_KEYS["llmed_text"]] = ""
                    st.session_state[SESSION_KEYS["translated_text"]] = ""
                    st.session_state[SESSION_KEYS["summary_text"]] = ""

                    if "chat_session" in st.session_state:
                        del st.session_state["chat_session"]
                    if "chat_display_history" in st.session_state:
                        st.session_state["chat_display_history"] = []
                else:
                    st.error(
                        "Failed to retrieve transcript. The video might not have captions or they are disabled."
                    )
        else:
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
                    # Reset LLM text and summary after crawling
                    st.session_state[SESSION_KEYS["llmed_text"]] = ""
                    st.session_state[SESSION_KEYS["translated_text"]] = ""
                    st.session_state[SESSION_KEYS["summary_text"]] = ""

                    if "chat_session" in st.session_state:
                        del st.session_state["chat_session"]
                    if "chat_display_history" in st.session_state:
                        st.session_state["chat_display_history"] = []
                except (ClientConnectorError, InvalidURL):
                    st.error(
                        f"Failed to connect to URL: {url}. Please check the URL and your network connection."
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    st.sidebar.radio(
        "View Mode",
        ["Crawled", "AI Processed", "Translated", "Summary", "Chatbot"],
        key="view_mode",
    )

    if st.sidebar.button("Show Markdown Code", width="stretch"):
        mode = st.session_state.get("view_mode", "Crawled")
        text_to_show = ""
        if mode == "Crawled":
            text_to_show = st.session_state.get(SESSION_KEYS["crawled_text"])
        elif mode == "AI Processed":
            text_to_show = st.session_state.get(SESSION_KEYS["llmed_text"])
        elif mode == "Translated":
            text_to_show = st.session_state.get(SESSION_KEYS["translated_text"])
        elif mode == "Summary":
            text_to_show = st.session_state.get(SESSION_KEYS["summary_text"])
        elif mode == "Chatbot":
            chat_history = st.session_state.get("chat_display_history", [])
            assistant_messages = [
                m["content"] for m in chat_history if m["role"] == "assistant"
            ]
            if assistant_messages:
                text_to_show = assistant_messages[-1]
            else:
                st.sidebar.warning("No assistant answer to show.")
                text_to_show = None
        else:
            text_to_show = None

        if text_to_show:
            show_markdown_code(text_to_show)
        elif mode != "Chatbot":
            st.sidebar.warning(f"No content in '{mode}' to show.")

    mode = st.session_state.get("view_mode", "Crawled")
    text_to_download = ""
    file_name = "download.pdf"

    if mode == "Crawled":
        text_to_download = st.session_state.get(SESSION_KEYS["crawled_text"])
        file_name = "crawled.pdf"
    elif mode == "AI Processed":
        text_to_download = st.session_state.get(SESSION_KEYS["llmed_text"])
        file_name = "ai_processed.pdf"
    elif mode == "Translated":
        text_to_download = st.session_state.get(SESSION_KEYS["translated_text"])
        file_name = "translated.pdf"
    elif mode == "Summary":
        text_to_download = st.session_state.get(SESSION_KEYS["summary_text"])
        file_name = "summary.pdf"
    elif mode == "Chatbot":
        chat_history = st.session_state.get("chat_display_history", [])
        assistant_messages = [
            m["content"] for m in chat_history if m["role"] == "assistant"
        ]
        if assistant_messages:
            text_to_download = assistant_messages[-1]
            file_name = "chatbot_answer.pdf"
        else:
            text_to_download = None
    else:
        text_to_download = None

    if text_to_download:
        pdf_data = download_pdf(text_to_download)
        if pdf_data:
            st.sidebar.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=file_name,
                mime="application/pdf",
                width="stretch",
            )


def render_main_content():
    """Renders the main content area."""
    mode = st.session_state.get("view_mode", "Crawled")

    crawled_text = st.session_state.get(SESSION_KEYS["crawled_text"])
    llmed_text = st.session_state.get(SESSION_KEYS["llmed_text"])
    translated_text = st.session_state.get(SESSION_KEYS["translated_text"])
    summary_text = st.session_state.get(SESSION_KEYS["summary_text"])

    if mode == "Crawled":
        if crawled_text:
            st.markdown(crawled_text)
        else:
            st.info("No crawled content. Enter a URL and click the 'Crawl' button.")

    elif mode == "AI Processed":
        if crawled_text and not llmed_text:
            with st.spinner("Extracting main content with Gemini..."):
                llmed_text = convert_by_gemini(PROMPTS["extraction"], crawled_text)
                st.session_state[SESSION_KEYS["llmed_text"]] = llmed_text

        if llmed_text:
            st.markdown(llmed_text)
        else:
            st.info("No processed content. Please run the crawler first.")

    elif mode == "Translated":
        if crawled_text and not llmed_text:
            with st.spinner("Extracting main content with Gemini..."):
                llmed_text = convert_by_gemini(PROMPTS["extraction"], crawled_text)
                st.session_state[SESSION_KEYS["llmed_text"]] = llmed_text

        if llmed_text and not translated_text:
            with st.spinner("Translating content with Gemini..."):
                translated_text = convert_by_gemini(PROMPTS["translation"], llmed_text)
                st.session_state[SESSION_KEYS["translated_text"]] = translated_text
                st.session_state[SESSION_KEYS["summary_text"]] = (
                    ""  # Reset summary to force re-summarization
                )

        if translated_text:
            st.markdown(translated_text)
        else:
            st.info("No translated content.")

    elif mode == "Summary":
        if crawled_text and not llmed_text:
            with st.spinner("Extracting main content with Gemini..."):
                llmed_text = convert_by_gemini(PROMPTS["extraction"], crawled_text)
                st.session_state[SESSION_KEYS["llmed_text"]] = llmed_text

        # Use translated text if available, otherwise use AI processed text
        input_text = translated_text if translated_text else llmed_text

        if input_text and not summary_text:
            with st.spinner("Generating summary with Gemini..."):
                summary_text = convert_by_gemini(PROMPTS["summary"], input_text)
                st.session_state[SESSION_KEYS["summary_text"]] = summary_text

        if summary_text:
            st.markdown(summary_text)
        else:
            st.info("No summarized content.")

    elif mode == "Chatbot":
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
