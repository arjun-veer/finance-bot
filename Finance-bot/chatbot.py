import streamlit as st
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import pandas as pd

# File to store user data and chat history
DATA_FILE = "user_data.json"
CHAT_HISTORY_FILE = "chat_history.json"

# Load environment variables
load_dotenv()

# Ensure chat history file exists
if not os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump([], file)

# Load or initialize user data
def load_user_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    return {"expenses": [], "budgets": {}, "income": 0, "savings": 0, "debts": 0}

def save_user_data(data):
    with open(DATA_FILE, "w") as file:
        json.dump(data, file)

# Load or initialize chat history
def load_chat_history():
    with open(CHAT_HISTORY_FILE, "r") as file:
        return json.load(file)

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file)

# Add an expense
def add_expense(data, category, amount, description):
    data["expenses"].append({
        "category": category,
        "amount": amount,
        "description": description
    })
    save_user_data(data)

# Set a budget
def set_budget(data, category, amount):
    data["budgets"][category] = amount
    save_user_data(data)

# Update additional financial data
def update_financial_data(data, income, savings, debts):
    data["income"] = income
    data["savings"] = savings
    data["debts"] = debts
    save_user_data(data)

# Get financial tips or answer user questions using Gemini API
def get_financial_tips_with_gemini(user_data, user_question=None, context=None):
    if context is None:  # Generate context only for the first interaction
        summary = calculate_summary(user_data)
        summary_text = "\n".join([f"Category: {cat}, Spent: {spent}" for cat, spent in summary.items()])
        additional_data = f"Income: {user_data['income']}, Savings: {user_data['savings']}, Debts: {user_data['debts']}"
        context = f"Financial Summary:\n{summary_text}\nAdditional Data:\n{additional_data}"

    if user_question:
        prompt_template = """
        Based on the following context, answer the user's question. If the question is not related to finance, respond with "Ask questions related to finance":
        Context:
        {context}
        Question: {question}
        """
        input_variables = ["context", "question"]
        inputs = {"context": context, "question": user_question}
   

    # Configure Gemini API
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run(**inputs)
    return response, context

# Calculate spending summary
def calculate_summary(data):
    summary = {}
    for expense in data["expenses"]:
        category = expense["category"]
        summary[category] = summary.get(category, 0) + expense["amount"]
    return summary

# Main chatbot interface
def main():
    st.set_page_config(page_title="Finance Assistant Chatbot", layout="wide")
    st.title("Finance Assistant Chatbot 💰")

    # Sidebar menu
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox("Choose an option", ["Track Expenses", "Set Budgets", "Update Financial Data", "View Summary", "Get Financial Tips"])

    categories = ["Food", "Education Fees", "Entertainment", "Utilities", "Healthcare", "Other"]

    if option == "Track Expenses":
        st.header("Track Expenses")
        category = st.selectbox("Category", categories)
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        description = st.text_area("Description")
        if st.button("Add Expense"):
            if category and amount > 0:
                add_expense(load_user_data(), category, amount, description)
                st.success("Expense added successfully!")
            else:
                st.error("Please provide valid category and amount.")

    elif option == "Set Budgets":
        st.header("Set Budgets")
        category = st.selectbox("Category", categories)
        amount = st.number_input("Budget Amount", min_value=0.0, step=0.01)
        if st.button("Set Budget"):
            if category and amount > 0:
                set_budget(load_user_data(), category, amount)
                st.success("Budget set successfully!")
            else:
                st.error("Please provide valid category and amount.")

    elif option == "Update Financial Data":
        st.header("Update Financial Data")
        income = st.number_input("Monthly Income", min_value=0.0, step=0.01)
        savings = st.number_input("Total Savings", min_value=0.0, step=0.01)
        debts = st.number_input("Total Debts", min_value=0.0, step=0.01)
        if st.button("Update Data"):
            update_financial_data(load_user_data(), income, savings, debts)
            st.success("Financial data updated successfully!")

    elif option == "View Summary":
        st.header("Spending Summary")
        summary = calculate_summary(load_user_data())
        if summary:
            summary_df = pd.DataFrame([
                {"Category": category, "Spent": total, "Budget": load_user_data()["budgets"].get(category, "Not Set"), 
                 "Remaining": load_user_data()["budgets"].get(category, 0) - total if category in load_user_data()["budgets"] else "N/A"}
                for category, total in summary.items()
            ])
            st.table(summary_df)
        else:
            st.write("No expenses recorded yet.")

    elif option == "Get Financial Tips":
        st.header("Financial Tips")

        # Load chat history and context
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "context" not in st.session_state:
            st.session_state.context = None

        # Display the latest response dynamically
        latest_response_container = st.container()
        if st.session_state.chat_history:
            latest_entry = st.session_state.chat_history[-1]
            with latest_response_container:
                st.markdown(
                    f"""
                    <div style='background-color:#f8f9fa; padding:10px; margin-bottom:10px; border-radius:10px;'>
                        <div style='background-color:#d1e7dd; padding:10px; border-radius:10px; text-align:right; margin-bottom:5px;'>
                            <strong>You:</strong> {latest_entry['question']}
                        </div>
                        <div style='background-color:#f0f0f0; padding:10px; border-radius:10px; text-align:left;'>
                            <strong>Bot:</strong> {latest_entry['response']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Chat input dynamically at the end of the page
        st.markdown("---")
        if "user_question" not in st.session_state:
            st.session_state.user_question = ""

        # Display the input box at the end of the page
        user_question = st.text_input("Ask a question related to finance", value=st.session_state.user_question, key="chat_input")
        submit_button = st.button("Submit Question")

        if submit_button and user_question.strip():
            # Generate the bot's response
            response, context = get_financial_tips_with_gemini(load_user_data(), user_question, st.session_state.context)

            # Update context only if it's the first interaction
            if st.session_state.context is None:
                st.session_state.context = context

            # Add the user's question and bot's response to the chat history
            st.session_state.chat_history.append({"question": user_question, "response": response})
            save_chat_history(st.session_state.chat_history)

            # Update the placeholder response dynamically
            with latest_response_container:
                st.markdown(
                    f"""
                    <div style='background-color:#f8f9fa; padding:10px; margin-bottom:10px; border-radius:10px;'>
                        <div style='background-color:#d1e7dd; padding:10px; border-radius:10px; text-align:right; margin-bottom:5px;'>
                            <strong>You:</strong> {user_question}
                        </div>
                        <div style='background-color:#f0f0f0; padding:10px; border-radius:10px; text-align:left;'>
                            <strong>Bot:</strong> {response}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Clear the input box
            st.session_state.user_question = ""

if __name__ == "__main__":
    main()
