import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Predefined questions and answers about Thoughtful AI
qa_data = {
    "What does the eligibility verification agent (EVA) do?": 
        "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections.",
    "What does the claims processing agent (CAM) do?": 
        "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements.",
    "How does the payment posting agent (PHIL) work?": 
        "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden.",
    "Tell me about Thoughtful AI's Agents.": 
        "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others.",
    "What are the benefits of using Thoughtful AI's agents?": 
        "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
}

# Load the pre-trained model and tokenizer (DialoGPT)
model_name = "microsoft/DialoGPT-medium"  # Small model to keep things lightweight
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fallback response using the transformer model
def generate_llm_response(user_input):
    # Tokenize the input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Generate a response using the model
    response_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the response
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit app setup
st.title("Thoughtful AI - Customer Support Agent")

# Get user input
user_input = st.text_input("Ask a question about Thoughtful AI's Agents:")

# If the user submits a question
if user_input:
    # First check if the input is in predefined responses
    if user_input in qa_data:
        response = qa_data[user_input]
    else:
        response = generate_llm_response(user_input)
    
    st.write(f"**Answer:** {response}")
