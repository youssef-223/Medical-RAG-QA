# Medical RAG-QA: Retrieval-Augmented Generation for Medical and Clinical Knowledge

This project implements a **Retrieval-Augmented Generation (RAG)** architecture to answer medical and clinical questions based on a given dataset. It combines the power of **retrieval** to fetch relevant context with **generation** to produce coherent and accurate answers.

The dataset used in this project contains two main fields:
- **page_title**: The title of the source document.
- **page_text**: The detailed content of the document, from which context is retrieved.

The system is designed to retrieve the most relevant text passages and use them to answer questions, providing a helpful and user-friendly interface for medical professionals, researchers, and enthusiasts.

---

## Features

- **RAG Architecture**: Combines context retrieval with natural language generation for accurate answers.
- **Streamlit Interface**: A user-friendly interface to input queries and get real-time answers.
- **Source Document Display**: Allows users to view the documents used for generating the answers.
- **Customizable Backend**: Flexible pipeline that can be extended to other datasets or domains.

---

## How It Works

1. **Question Input**: The user inputs a medical or clinical question via the Streamlit interface.
2. **Context Retrieval**: The RAG pipeline retrieves the most relevant passages from the dataset using retrieval models.
3. **Answer Generation**: The retrieved context is passed to a language model, which generates a coherent answer.
4. **Source Documents**: Users can view the source documents for transparency and validation.

---

## Key Components

- **Streamlit Frontend**: Simplifies interaction with the system through a web interface.
- **NGROK Deployment**: Enables hosting and sharing the app via public URLs.
- **Regex Answer Extraction**: Extracts concise, helpful answers from model-generated results.
- **Error Handling**: Handles edge cases gracefully, providing feedback to users.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/youssef-223/Medical-RAG-QA.git
   cd Medical-RAG-QA
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. (Optional) Deploy using NGROK

    1. Install NGROK: [Download here](https://ngrok.com/download).
    2. Run the following command:
     ```bash
     ngrok http 8501
     ```
---

## Dataset

The dataset contains two columns:

- **`page_title`**: Title of the document.
- **`page_text`**: The text body used for answering questions.

The context is retrieved from `page_text` to generate answers.

---

## Usage

1. Run the app using the Streamlit command.
2. Enter your question in the input box.
3. Click the **Get Answer** button to fetch an answer.
4. If enabled, view the source documents used for generating the answer.

---

## Example

### Query:

_What are other names for Paracetamol?_

### Generated Answer:

_Other names for paracetamol include Tylenol, Panadol, and Nuprin._

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push:
   ```bash
    git push origin feature-name
   ```
4. Create a pull request.

---

### License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Contact
For any questions or support, feel free to reach out:

GitHub Issues:[Open an issue](https://github.com/youssef-223/Medical-RAG-QA/issues)
