import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Load, chunk and index the contents of the blog.
# bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs={"parse_only": bs_strainer},
# )
loader = DirectoryLoader("/Users/terryblanc/Documents/LigonierStudy")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=1024)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="orca-mini", show_progress=True,num_ctx=4096))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = Ollama(model="orca-mini", num_ctx=4096)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# for chunk in rag_chain_with_source.stream("What is Task Decomposition"):
#     print(chunk)

output = {}
curr_key = None
for chunk in rag_chain_with_source.stream("What is theology?"):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key != curr_key:
            print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
        else:
            print(chunk[key], end="", flush=True)
        curr_key = key

# with open('C:\\Users\\tblanc\\Downloads\\ddowney_log_chat_output.txt', 'w') as file:
#     file.write(str(output["answer"]))