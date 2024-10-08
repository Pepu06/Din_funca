from deepeval import evaluate
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
import os
from dotenv import load_dotenv
import google.generativeai as genai
from generatyresponsychan import generate_response

# Load environment variables
load_dotenv()

# Set up your Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Define your test cases
test_cases = [
    {
        "input": "What if these shoes don't fit?",
        "expected_output": "You are eligible for a 30 day full refund at no extra cost.",
        "retrieval_context": ["All customers are eligible for a 30 day full refund at no extra cost."]
    },
    # Add more test cases as needed
]

# Prepare LLMTestCase instances
llm_test_cases = []
for case in test_cases:
    actual_output = generate_response(case["input"])  # Generate output using Gemini
    llm_test_cases.append(LLMTestCase(
        input=case["input"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"]
    ))

# Define the metric
metric = RagasMetric(threshold=0.5, model="gemini-pro")

# Evaluate the test cases
scores = evaluate(llm_test_cases, [metric])

# Print results
for test_case, score in zip(llm_test_cases, scores):
    print(f"Input: {test_case.input}")
    print(f"Actual Output: {test_case.actual_output}")
    print(f"Expected Output: {test_case.expected_output}")
    print(f"Score: {score}")
