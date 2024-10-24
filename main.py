
import os
import json
import streamlit as st
from streamlit_tree_select import tree_select

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import get_upload_dir_content
from doc_indexer.docIndexer import *

with open("./config.json", "r") as f:
    config = json.load(f)

os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")
GOOGLE_API_KEY = config.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
working_dir = "uploaded_pdfs"
local_vector_store = "chroma_tiger_db"
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=local_vector_store,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def create_rag_chain(model, retriever):
    return create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(
            model,
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "you are an assistant for question answering task."
                        "you always generate nice detailed answer with explanation from the context."
                        "Only use the following context to answer the user question."
                        "if you do not know the answer say that the retrieved context is not sufficient to generate the answer."
                        "\n\n\n"
                        "CONTEXT"
                        "\n"
                        "{context}",
                    ),
                    ("human", "{input}"),
                ]
            ),
        ),
    )


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = create_rag_chain(model=model, retriever=retriever)



def main():
    st.set_page_config("Chat PDF")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.header(":robot_face: :blue[Chat with Your PDFs]")
    with col2:
        st.image(
            "https://miro.medium.com/v2/resize:fit:1056/1*ta51INqM9SxFrY5yX5ajJg.png",
            width=150,
        )
    st.divider()

    with st.sidebar:
        st.image(
            "https://www.tigeranalytics.com/wp-content/uploads/2023/09/TA-Logo-resized-for-website_.png"
        )
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Vectorize & Process Button",
            accept_multiple_files=True,
        )
        for uploaded_file in pdf_docs:
            bytes_data = uploaded_file.read()
            out_path = os.path.join(working_dir, uploaded_file.name)
            if not os.path.exists(out_path):
                with open(out_path, mode="wb") as of:
                    of.write(bytes_data)

        selected_files = tree_select([get_upload_dir_content(working_dir)])
        if st.button("Vectorize & Process", use_container_width=True):
            pdf_files = [
                x for x in selected_files.get("checked") if x.lower().endswith(".pdf")
            ]
            with st.spinner("Processing..."):
                docx_ids = []
                docx = []
                for ip, p in enumerate(pdf_files):
                    ip = f"{(ip+1)}"
                    page_list = convert_pdf_to_text_description(pdf_file_path=p)
                    temp = []
                    for _, px in enumerate(page_list):
                        _, p_content = px
                        temp.append(
                            p_content
                        )
                    docx_ids.append(ip)
                    doc_content = "\n".join(temp)
                    d = Document(
                        page_content=doc_content,
                        metadata={
                            "source_file": os.path.basename(p),
                            "keywords": model.invoke(
                                f"extract all keywords from the following text, no heading please, generate only keywords \n\n {doc_content}"
                            ).content
                        },
                        id=ip,
                    )
                    docx.append(
                        d
                    )
                    st.markdown(f":green[processed] - {p}")
                vector_store.add_documents(documents=docx, ids=docx_ids)



    if user_question := st.chat_input("Ask a Question from the PDF Files"):
        st_callback = StreamlitCallbackHandler(st.container())
        st.markdown(user_question)
        results = st.session_state.rag_chain.invoke(
            {"input": user_question}, {"callback": [st_callback]}
        )
        answer = results.get("answer")
        context = results.get("context")
        st.markdown(answer)
        tabs = st.tabs(
            [f"Retrieved-{(i+1)}" for i in range(len(context))]
        )
        for ti, t in enumerate(tabs):
            with t:
                c1,  c3 = st.columns(2)
                with c1:
                    st.subheader(":blue[Extracted ]")
                    st.markdown(context[ti].page_content)
                with c3:
                    st.subheader(":blue[Original Content]")
                    img_dir = os.path.join(working_dir, context[ti].metadata.get("source_file").replace(".pdf", ""))
                    for f in os.listdir(img_dir):
                        st.image(os.path.join(img_dir, f))

if __name__ == "__main__":
    main()
