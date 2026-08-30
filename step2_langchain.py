# from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini")

response = llm.invoke(
    "Hello How are you? its Misgana"
)





print(response.content)


