import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
import os
# import dotenv



st.title("M.Hans Bot")



from langchain.tools import tool
from datetime import datetime

@tool
def time_tool(query: str) -> str:
    """
    A tool that tells the current time. Use this when a user asks for the current time.
    """
    current_date, current_time =  datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(" ")
    return f"The current date is {current_date} and the current time is {current_time}"

search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia, time_tool]
prompt = hub.pull("hwchase17/openai-functions-agent")


# dotenv.load_dotenv()

open_key = os.environ.get("OPENAI_API_KEY")
antro_key = os.environ.get("ANTHROPIC_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if st.button("clear chat"):
    st.session_state.memory = []
    st.session_state.chat_memory = []

st.session_state.selected_model = st.selectbox('choose model to use', ('gpt-4o-mini', 'claude', 'llama', 'mixtral'))

if st.session_state.selected_model == 'gpt-4o-mini':
    model = ChatOpenAI(model="gpt-4o-mini", api_key=open_key)

elif st.session_state.selected_model == 'claude':
    model = ChatAnthropic(model="claude-3-opus-20240229", api_key=antro_key)

elif st.session_state.selected_model == 'llama':
    model = ChatGroq(model='llama3-groq-70b-8192-tool-use-preview', api_key=groq_key)

elif st.session_state.selected_model == 'mixtral':
    model = ChatGroq(model='mixtral-8x7b-32768', api_key=groq_key)



model = model.bind_tools(tools)

chain = RunnablePassthrough.assign(
                agent_scratchpad = lambda x: format_to_tool_messages(x["intermediate_steps"])
                ) | prompt | model | ToolsAgentOutputParser()

agent_executor = AgentExecutor(agent=chain, tools=tools,
                                           handle_parsing_errors=True,
                                           return_intermediate_steps=True)



for chat in st.session_state.chat_memory:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])


if prompt:= st.chat_input("what is moving?"):
    with st.chat_message("user"):
        st.session_state.memory.append(HumanMessage(content=prompt))
        st.markdown(prompt)
        st.session_state.chat_memory.append({"role":"user", "message":prompt})

    with st.chat_message("ai"):
        # response = model.invoke(st.session_state.memory).content
        response = agent_executor.invoke({"input":prompt, 'chat_history': st.session_state.memory}, return_only_outputs = True)["output"]
        if st.session_state.selected_model == 'claude':
            response = response[0]['text'].split('</thinking>')[-1]
        st.markdown(response)
        st.session_state.memory.append(AIMessage(content=response))
        st.session_state.chat_memory.append({"role":"ai", "message":response})

