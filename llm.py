from langchain_openai import ChatOpenAI

from configs import MODEL_NAME, TEMPERATURE


model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
