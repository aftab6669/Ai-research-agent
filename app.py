import os
import streamlit as st

# Turn off CrewAI telemetry (optional, keeps things quiet)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

st.set_page_config(page_title="AI Research Agent", page_icon="🔎")
st.title("🔎 AI Research Agent")
st.write("Enter a topic and the agent will search the web and write a report.")


# ---------- 1. Search tool (DuckDuckGo, free) ----------
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


# ---------- 2. LLM (Groq via OpenAI-compatible API) ----------
def get_llm(api_key: str) -> LLM:
    return LLM(
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.3,
    )


# ---------- 3. Build the crew (one agent, one task) ----------
def run_research(topic: str, api_key: str) -> str:
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' thoroughly and write a clear, accurate report.",
        backstory=(
            "You are an experienced analyst who searches the web, checks "
            "multiple sources, and writes well-structured reports."
        ),
        tools=[duckduckgo_search],
        llm=get_llm(api_key),
        verbose=False,
        allow_delegation=False,
        max_iter=6,  # limits how many search/think loops it can do
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n"
            "Use the search tool 2-4 times with different queries. "
            "Then write a report in Markdown."
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
                report = run_research(topic.strip(), api_key)
                st.markdown(report)
                st.download_button(
                    "Download report (.md)", report, file_name="report.md"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
