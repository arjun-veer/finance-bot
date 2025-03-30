
# Finance Bot

This repository contains the Finance Bot, a project utilizing an LLM (Large Language Model) RAG (Retrieval-Augmented Generation) model using various LLM models. The bot is designed to provide financial insights and answer finance-related queries.


## Overview
The Finance Bot leverages a Retrieval-Augmented Generation approach to provide accurate and contextually relevant financial information. By integrating various LLM models, the bot can understand and generate human-like responses to a wide range of financial queries.

## Features
- **Accurate Financial Insights**: Uses advanced machine learning models to provide precise financial information.
- **Context-Aware Responses**: Understands the context of queries for more relevant answers.
- **Easy Integration**: Can be integrated into various platforms and applications.

## Installation

### Prerequisites
- Python 3.12 or higher
- conda (Anaconda or Miniconda)

### Create a Conda Environment
1. **Create a New Environment**:
   ```bash
   conda create -p venv python=3.12 -y
   ```

2. **Activate the Environment**:
   ```bash
   conda activate venv/
   ```

### Set Up API Keys
Create a `.env` file in the root directory and add your API keys.

### Install Dependencies
After activating the virtual environment, install the necessary dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Bot
To start the Finance Bot, run the following command:
```bash
streamlit run app.py
```

### Example Queries
- "What is the current stock price of Apple?"
- "Explain the concept of compound interest."
- "What are the latest trends in the cryptocurrency market?"


## Contributing
We welcome contributions to improve the Finance Bot. Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a Pull Request.


