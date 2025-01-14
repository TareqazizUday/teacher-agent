import streamlit as st
from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
from phi.tools.serpapi_tools import SerpApiTools
from phi.utils.pprint import pprint_run_response
import os

# Set page configuration
st.set_page_config(page_title="👨‍🏫 AI Teaching Agent Team", layout="centered")

# Initialize session state for API keys and topic
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''
if 'serpapi_api_key' not in st.session_state:
    st.session_state['serpapi_api_key'] = ''
if 'topic' not in st.session_state:
    st.session_state['topic'] = ''

# Streamlit sidebar for API keys
with st.sidebar:
    st.title("API Keys Configuration")
    st.session_state['openai_api_key'] = st.text_input("Enter your OpenAI API Key", type="password").strip()
    st.session_state['serpapi_api_key'] = st.text_input("Enter your SerpAPI Key", type="password").strip()
    
    # Add info about terminal responses
    st.info("Note: You can also view detailed agent responses\nin your terminal after execution.")

# Validate API keys
if not st.session_state['openai_api_key'] or not st.session_state['serpapi_api_key']:
    st.error("Please enter both OpenAI and SerpAPI keys in the sidebar.")
    st.stop()

# Set the OpenAI API key from session state
os.environ["OPENAI_API_KEY"] = st.session_state['openai_api_key']

# Create the Professor agent
professor_agent = Agent(
    name="Professor",
    role="Research and Knowledge Specialist", 
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[],  # No tools needed for local file saving
    instructions=[
        "Create a comprehensive knowledge base that covers fundamental concepts, advanced topics, and current developments of the given topic.",
        "Explain the topic from first principles first. Include key terminology, core principles, and practical applications and make it as a detailed report that anyone who's starting out can read and get maximum value out of it.",
        "Make sure it is formatted in a way that is easy to read and understand.",
        "Provide the response in plain text.",
    ],
    show_tool_calls=False,
    markdown=True,
)

# Create the Academic Advisor agent
academic_advisor_agent = Agent(
    name="Academic Advisor",
    role="Learning Path Designer",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[],  # No tools needed for local file saving
    instructions=[
        "Using the knowledge base for the given topic, create a detailed learning roadmap.",
        "Break down the topic into logical subtopics and arrange them in order of progression, a detailed report of roadmap that includes all the subtopics in order to be an expert in this topic.",
        "Include estimated time commitments for each section.",
        "Present the roadmap in a clear, structured format.",
        "Provide the response in plain text.",
    ],
    show_tool_calls=False,
    markdown=True
)

# Create the Research Librarian agent
research_librarian_agent = Agent(
    name="Research Librarian",
    role="Learning Resource Specialist",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[SerpApiTools(api_key=st.session_state['serpapi_api_key'])],
    instructions=[
        "Make a list of high-quality learning resources for the given topic.",
        "Use the SerpApi search tool to find current and relevant learning materials.",
        "Include technical blogs, GitHub repositories, official documentation, video tutorials, and courses.",
        "Present the resources in a curated list with descriptions and quality assessments.",
        "Provide the response in plain text.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Create the Teaching Assistant agent
teaching_assistant_agent = Agent(
    name="Teaching Assistant",
    role="Exercise Creator",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[SerpApiTools(api_key=st.session_state['serpapi_api_key'])],
    instructions=[
        "Create comprehensive practice materials for the given topic.",
        "Use the SerpApi search tool to find example problems and real-world applications.",
        "Include progressive exercises, quizzes, hands-on projects, and real-world application scenarios.",
        "Ensure the materials align with the roadmap progression.",
        "Provide detailed solutions and explanations for all practice materials.",
        "Provide the response in plain text.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Function to save content to a text file
def save_to_text_file(agent_name, content, topic):
    # Create a directory for the topic if it doesn't exist
    safe_topic = "".join([c if c.isalnum() or c in (' ', '_') else '_' for c in topic])
    os.makedirs(f"outputs/{safe_topic}", exist_ok=True)
    # Define the file path
    file_path = f"outputs/{safe_topic}/{agent_name}.txt"
    # Write the content to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

# Streamlit main UI
st.title("👨‍🏫 AI Teaching Agent Team")
st.markdown("Enter a topic to generate a detailed learning path and resources")

# Add info message about file outputs
st.info("📝 The agents will create detailed text files for each section (Professor, Academic Advisor, Research Librarian, and Teaching Assistant). You can download these files below after processing.")

# Query bar for topic input
st.session_state['topic'] = st.text_input("Enter the topic you want to learn about:", placeholder="e.g., Machine Learning, LoRA, etc.")

# Start button
if st.button("Start"):
    if not st.session_state['topic']:
        st.error("Please enter a topic.")
    else:
        topic = st.session_state['topic']
        # Display loading animations while generating responses
        with st.spinner("Generating Knowledge Base..."):
            professor_response: RunResponse = professor_agent.run(
                f"the topic is: {topic}"
            )
        
        with st.spinner("Generating Learning Roadmap..."):
            academic_advisor_response: RunResponse = academic_advisor_agent.run(
                f"the topic is: {topic}"
            )
        
        with st.spinner("Curating Learning Resources..."):
            research_librarian_response: RunResponse = research_librarian_agent.run(
                f"the topic is: {topic}"
            )
        
        with st.spinner("Creating Practice Materials..."):
            teaching_assistant_response: RunResponse = teaching_assistant_agent.run(
                f"the topic is: {topic}"
            )

        # Save responses to local text files
        professor_file = save_to_text_file("Professor", professor_response.content, topic)
        academic_advisor_file = save_to_text_file("Academic_Advisor", academic_advisor_response.content, topic)
        research_librarian_file = save_to_text_file("Research_Librarian", research_librarian_response.content, topic)
        teaching_assistant_file = save_to_text_file("Teaching_Assistant", teaching_assistant_response.content, topic)

        # Display download buttons for each file
        st.markdown("### Downloadable Text Files:")
        try:
            with open(professor_file, "rb") as f:
                professor_bytes = f.read()
            st.download_button(
                label="Download Professor Document",
                data=professor_bytes,
                file_name=f"{topic}_Professor.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error downloading Professor document: {e}")

        try:
            with open(academic_advisor_file, "rb") as f:
                academic_advisor_bytes = f.read()
            st.download_button(
                label="Download Academic Advisor Document",
                data=academic_advisor_bytes,
                file_name=f"{topic}_Academic_Advisor.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error downloading Academic Advisor document: {e}")

        try:
            with open(research_librarian_file, "rb") as f:
                research_librarian_bytes = f.read()
            st.download_button(
                label="Download Research Librarian Document",
                data=research_librarian_bytes,
                file_name=f"{topic}_Research_Librarian.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error downloading Research Librarian document: {e}")

        try:
            with open(teaching_assistant_file, "rb") as f:
                teaching_assistant_bytes = f.read()
            st.download_button(
                label="Download Teaching Assistant Document",
                data=teaching_assistant_bytes,
                file_name=f"{topic}_Teaching_Assistant.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error downloading Teaching Assistant document: {e}")

        # Optionally, display responses in the Streamlit UI using pprint_run_response
        st.markdown("### Professor Response:")
        st.text(professor_response.content)
        pprint_run_response(professor_response, markdown=True)
        st.divider()
        
        st.markdown("### Academic Advisor Response:")
        st.text(academic_advisor_response.content)
        pprint_run_response(academic_advisor_response, markdown=True)
        st.divider()

        st.markdown("### Research Librarian Response:")
        st.text(research_librarian_response.content)
        pprint_run_response(research_librarian_response, markdown=True)
        st.divider()

        st.markdown("### Teaching Assistant Response:")
        st.text(teaching_assistant_response.content)
        pprint_run_response(teaching_assistant_response, markdown=True)
        st.divider()

# Information about the agents
st.markdown("---")
st.markdown("### About the Agents:")
st.markdown("""
- **Professor**: Researches the topic and creates a detailed knowledge base.
- **Academic Advisor**: Designs a structured learning roadmap for the topic.
- **Research Librarian**: Curates high-quality learning resources.
- **Teaching Assistant**: Creates practice materials, exercises, and projects.
""")
