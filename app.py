import os
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Turn off CrewAI telemetry (optional, keeps things quiet)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

st.set_page_config(page_title="AI Research Agent", page_icon="🔎")
st.title("🔎 AI Research Agent")
st.write("Enter a topic and the agent will search the web and write a report.")


# ---------- 1. Tools ----------
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input is a search query string.
    Returns titles, links and short snippets of the top results."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        # Trim snippets so we stay within Groq's free token limits
        lines.append(f"- {r['title']}\n  {r['href']}\n  {r['body'][:300]}")
    return "\n".join(lines)


@tool("Read Webpage")
def read_webpage(url: str) -> str:
    """Open a web page URL and return its main text (shortened).
    Input is a full URL starting with http:// or https://."""
    try:
        resp = requests.get(
            url.strip().strip("[]\"'"),
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:3000] if text else "The page had no readable text."
    except Exception as e:
        return f"Could not read the page: {e}"


# ---------- 2. LLM (Groq via OpenAI-compatible API) ----------
def get_llm(api_key: str) -> LLM:
    return LLM(
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.2,
    )


# ---------- 3. Build the crew (one agent, one task) ----------
def run_research(topic: str, api_key: str) -> str:
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' thoroughly and write a clear, accurate report.",
        backstory=(
            "You are an experienced analyst who searches the web, checks "
            "multiple sources, and writes well-structured reports. "
            "You have ONLY two tools: 'DuckDuckGo Search' and 'Read Webpage'. "
            "Never call any other tool such as open_file, browser or python."
        ),
        tools=[duckduckgo_search, read_webpage],
        llm=get_llm(api_key),
        verbose=False,
        allow_delegation=False,
        max_iter=8,  # limits how many search/think loops it can do
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n"
            "Use 'DuckDuckGo Search' 2-4 times with different queries. "
            "You may use 'Read Webpage' on at most 2 URLs from the results. "
            "Do not use any other tools. Then write a report in Markdown."
        ),
        expected_output=(
            "A Markdown report with: a title, an introduction, 3-5 key "
            "sections with headings, a short conclusion, and a 'Sources' "
            "list of URLs you actually used."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff(inputs={"topic": topic})
    return result.raw


def run_with_retry(topic: str, api_key: str, attempts: int = 2) -> str:
    """Try again if the model calls a tool that doesn't exist."""
    last_error = None
    for _ in range(attempts):
        try:
            return run_research(topic, api_key)
        except Exception as e:
            last_error = e
            if "tool_use_failed" not in str(e) and "not in request.tools" not in str(e):
                raise
    raise last_error


# ---------- 4. Streamlit UI ----------
# The API key is read ONLY from Streamlit secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. In Streamlit Cloud, open your app's "
        "Settings > Secrets and add:  GROQ_API_KEY = \"your_key_here\""
    )
    st.stop()

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Impact of fintech on banking in Pakistan",
)

if st.button("Generate report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Researching and writing... this can take 1-2 minutes."):
            try:
                report = run_with_retry(topic.strip(), api_key)
                st.markdown(report)
                st.download_button(
                    "Download report (.md)", report, file_name="report.md"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
