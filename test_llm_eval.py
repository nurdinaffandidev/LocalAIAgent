# Import LangChain modules for QA generation and evaluation
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM  # Ollama model integration
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from vector import retriever  # Custom retriever logic
import pandas as pd

# ---------- Step 1: Load and Prepare Dataset ----------

# Load a sample of restaurant reviews from CSV
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Convert rows into LangChain Document format
documents = [
    Document(
        page_content=row["Title"] + " " + row["Review"],  # Concatenate title and review as content
        metadata={"rating": row["Rating"], "date": row["Date"]}
    )
    for _, row in df.sample(5).iterrows()  # Random sample of 5 reviews
]

# ---------- Step 2: Generate QA Pairs ----------

# Load Ollama model (assumes llama3.2 is available locally)
model = OllamaLLM(model="llama3.2")

# Create a QA generation chain using the LLM
qna_generator = QAGenerateChain.from_llm(model)

# Run batch generation of QA pairs for each document
raw_outputs = qna_generator.batch(documents)

# Reformat output into a list of QA examples
qa_examples = []
for output, doc in zip(raw_outputs, documents):
    qa_pair = output["qa_pairs"]  # Extract generated question and answer
    qa_examples.append({
        "query": qa_pair["query"],
        "answer": qa_pair["answer"],
        "context": doc.page_content
    })

# ---------- Step 3: Use Retriever to Predict Answer ----------

# Define the QA answering prompt template
qa_prompt = ChatPromptTemplate.from_template("""
You are an expert on pizza restaurants.
Given the reviews: {reviews}
Answer this question: {query}
""")

# Define a custom QA chain that gets reviews using retriever and passes to LLM
def custom_chain(example):
    reviews = retriever.invoke(example["query"])  # Use vector retriever to fetch relevant reviews
    response = model.invoke(qa_prompt.invoke({
        "reviews": reviews,
        "query": example["query"]
    }))
    return {
        "query": example["query"],
        "answer": example["answer"],
        "context": example["context"],
        "prediction": response  # Store model-generated answer
    }

# Run the chain for all examples to get predictions
predictions = [custom_chain(example) for example in qa_examples]

# ---------- Step 4: Define Output Format for Grading ----------

# Define the required output schema for the grading result
response_schemas = [
    ResponseSchema(name="result", description="Is the answer CORRECT or INCORRECT"),
    ResponseSchema(name="comment", description="Justification of the grade")
]

# Create a parser to enforce structured output
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# ---------- Step 5: Prompt for Grading ----------

# Grading prompt instructs the model to evaluate predicted answers
grading_prompt = ChatPromptTemplate.from_template("""
You are a fair and critical grader.

Question: {query}
Reference Answer: {answer}
Predicted Answer: {prediction}

Evaluate the predicted answer. Is it CORRECT or INCORRECT? Explain your reasoning.

{format_instructions}
""".strip())  # Adds structured format instructions from parser

# Custom evaluation function that applies the grading prompt and parses the output
def custom_eval(example):
    formatted_prompt = grading_prompt.format_prompt(
        query=example["query"],
        answer=example["answer"],
        prediction=example["prediction"],
        format_instructions=parser.get_format_instructions()  # Tells the model to return JSON
    )
    raw_response = model.invoke(formatted_prompt)  # Call the LLM
    return parser.parse(raw_response)  # Convert raw text into structured output

# Run grading over all predictions
graded_results = [custom_eval(pred) for pred in predictions]

# ---------- Step 6: Print Final Evaluation ----------

# Display each example, prediction, and grading result
for i, result in enumerate(graded_results):
    print(f"\nExample {i+1}")
    print("Question:", predictions[i]["query"])
    print("Expected Answer:", predictions[i]["answer"])
    print("Model Answer:", predictions[i]["prediction"])
    print("Grade:", result["result"])      # CORRECT / INCORRECT
    print("Comments:", result["comment"])  # Explanation from the model
    print("=" * 50)
