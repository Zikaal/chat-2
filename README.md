# Advanced Programming Assignment 2: Document-based Q&A System

## Description
This project is a browser-based application that allows users to upload documents and ask questions about their content. Powered by a custom-built Q&A system, the application efficiently processes uploaded files and provides answers within the context of the documents. It supports single or multiple file uploads and is designed for modular and extensible functionality.

---

## Features
- Browser-based chat functionality using Streamlit.
- Integration with the **Ollama** large language model.
- Storage of user queries and responses in **MongoDB**.
- **File Uploads:** Attach one or multiple documents simultaneously.
- **Contextual Q&A:** Ask questions and receive answers based on uploaded document content.
- **Expandable Format Support:** Works with `.txt` files by default but can be extended to handle other formats.
- **Streamlit Integration:** Browser-based user interface for seamless interaction.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- **Ollama LLM** installed locally 
- A running instance of **MongoDB** 

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Zikaal/chat-2.git
   cd chat-2
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
- Create a `.env` file in the root directory if needed for additional configurations.

---

## Usage

1. Run the application:
   ```bash
   streamlit run src/chatbot.py
   ```

2. Open your browser at the provided URL.

3. Upload one or multiple documents using the file upload interface.

4. Ask questions about the uploaded documents in the input field. The system will provide contextually relevant answers.

---

## Examples

### Example 1:
- **Uploaded Document:** A `.txt` file containing the project requirements.
- **User Query:** "What is the grading policy?"
- **Response:** "The grading policy includes 35 points for file attachment functionality, 50 points for context quality, and 15 points for the README."

### Example 2:
- **Uploaded Document:** A `.txt` file about artificial intelligence.
- **User Query:** "What is the importance of AI in modern society?"
- **Response:** "AI plays a critical role in modern society by automating tasks, improving decision-making, and driving technological innovation."

---

## Project Structure
```bash
project-name/
│
├── README.md            # Documentation
├── requirements.txt     # Project dependencies
├── src/                 # Source code
│   ├── chatbot.py               # Main Streamlit application
├── test/                 
│   ├── chatbot.py          # Test cases for the application
└── .gitignore           # Ignored files
```

---

## License
- This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements
- Streamlit — for providing a simple web application framework.
- Ollama — for their powerful language model.
- MongoDB — for a reliable database solution.

