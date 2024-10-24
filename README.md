## PDF Chatbot
This project allows you to interact with PDFs using a chatbot interface. It leverages various technologies like OpenAI embeddings, Chroma for vector storage, and Google Generative AI for answering questions. Users can upload PDF files, process them, and ask questions to retrieve context-based answers from the documents.


## How to Run  (Python 3.9.6)

1. Create a virtual env with Python 3.9.6
2. Download the repo from git and go to the project dir
3. Create an OpenAI API key and put it in ´config.json´ under "OPENAI_API_KEY"
4. Create a Gemini API key and put it in ´config.json´ under "GOOGLE_API_KEY"
5. Run  -  ´pip install -r requirement.txt´
6. Install poppler and add installation dir path to PATH variable. You can visit https://stackoverflow.com/questions/53481088/poppler-in-path-for-pdf2image for detail.
7. Go to the project dir run - ´streamlit run main.py´