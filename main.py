# Import the OllamaLLM wrapper — lets you use LLMs running via Ollama in LangChain
from langchain_ollama.llms import OllamaLLM
# Import the ChatPromptTemplate to structure prompt messages with variables
from langchain_core.prompts import ChatPromptTemplate
# Import a retriever (likely a vector retriever you defined in `vector.py`)
# This is responsible for fetching relevant context (e.g., restaurant reviews)
from vector import retriever


# specify model
model = OllamaLLM(model="llama3.2")

# Define the system-level instruction (sets model behavior or personality)
# This tells the model to act as a pizza critique in this pizza restaurant.
system_template = """
You are an expert in answering about a pizza restaraunt.
Here are some relevant reviews: {reviews}
Here are some questions to answer: {question} 
"""

# Create a chat prompt template using the `system_template`
prompt = ChatPromptTemplate.from_template(system_template)
# Chain together: prompt → model → output
chain = prompt | model

def get_bool_input(prompt_message):
    while True:
        input_val = input(prompt_message).strip().lower()
        if input_val in ('yes', 'y'):
            return True
        elif input_val in ('no', 'n'):
            return False
        else:
            print("Please enter y/n: ")


def run_ai_agent(is_stream):
    while True:
        print("\n----------------------------------")
        question = input("Ask your question (/q to quit): ")
        if question == "/q":
            print("Goodbye 👋")
            break

        # Use the retriever (likely vector-based) to get relevant reviews
        reviews = retriever.invoke(question)

        if is_stream:
            # Use streams
            print("Result: (with streaming)")
            print("==================================")
            result = chain.stream({  # result type = <class 'generator'>
                "reviews": reviews,
                "question": question  # Eg.: "What is the best pizza in town?"
            })
            for chunk in result:
                print(chunk, end="", flush=True)
        else:
            result = chain.invoke({
                "reviews": reviews,
                "question": question  # Eg.: "What is the best pizza in town?"
            })
            print("Result: (no streaming)")
            print("==================================")
            print(result, end="\n\n")


if __name__ == "__main__":
    is_stream = get_bool_input("Use streaming? (y/n): ")
    run_ai_agent(is_stream)