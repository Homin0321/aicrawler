# AI Web Crawler

This is a simple web crawler application that uses AI to extract the main content from a webpage and summarize it.

## Features

- Crawls a given URL and converts the HTML content to Markdown.
- Uses Google's Gemini model to extract the main content from the crawled text.
- Summarizes the extracted content using the Gemini model.
- Displays the raw crawled content, the AI-processed content, the Markdown version of the processed content, and the summary in a user-friendly interface.

## Setup and Usage

### Prerequisites

- Python 3.7+
- Pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Homin0321/aicrawler.git
    cd aicrawler
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** by copying the example file:

    ```bash
    cp .env.example .env
    ```

2.  **Add your Gemini API key** to the `.env` file:

    ```
    GEMINI_API_KEY="your_gemini_api_key"
    ```

### Running the Application

Once you have completed the setup, you can run the application using Streamlit:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Dependencies

This project uses the following libraries:

-   `streamlit`: For creating the web application interface.
-   `crawl4ai`: For crawling the web pages.
-   `python-dotenv`: For managing environment variables.
-   `aiohttp`: Asynchronous HTTP client/server framework.
-   `google-generativeai`: For interacting with the Google Gemini API.
