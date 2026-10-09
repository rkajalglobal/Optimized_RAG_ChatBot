import os
import time
import hashlib
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

load_dotenv()


BETTERHELP_URLS = [
    "https://www.betterhelp.com/",
    "https://www.betterhelp.com/advice/insurance/how-many-therapy-sessions-does-insurance-cover/",
    "https://www.betterhelp.com/advice/therapy/do-we-need-couples-therapy/",
    "https://www.betterhelp.com/advice/psychology/what-is-the-mind-body-connection/",
    "https://www.betterhelp.com/reviews/"

]


PINECONE_INDEX_NAME = "betterhelp-rag"
PINECONE_NAMESPACE = "betterhelp-website"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # must match the embedding model above
PINECONE_API_KEY="pcsk_6iDZvQ_QA2yQZp7E1WDNVqiqA5BZCpk8MJCUPXFbUHjKdoxkZJFeH6Hc1FbdgQNB5Ci4yo"
OPENAI_API_KEY="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"


def demo_chatbot():
    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Create a .env file with: "
            "OPENAI_API_KEY=your_key_here"
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key
    )


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def get_pinecone_index():
    """Connect to the Pinecone index, creating it on first run."""
    api_key = PINECONE_API_KEY
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found. Add it to your .env file: "
            "PINECONE_API_KEY=your_key_here"
        )

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)

def namespace_vector_count(index):
    """Number of vectors already stored in our namespace (0 if none)."""
    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    ns = namespaces.get(PINECONE_NAMESPACE)
    return ns.vector_count if ns else 0


def make_chunk_ids(chunks):
    """
    Deterministic IDs (source URL + chunk position), so re-indexing
    overwrites existing records instead of creating duplicates.
    """
    ids, counters = [], {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        i = counters.get(source, 0)
        counters[source] = i + 1
        ids.append(hashlib.md5(f"{source}::{i}".encode()).hexdigest())
    return ids


def scrape_with_playwright(urls):
    """
    Uses Playwright headless browser to load pages, pass JS challenges,
    and convert pages into LangChain Document objects.
    """
    documents = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Setup real browser environment headers
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for url in urls:
            try:
                print(f"Scraping safely via Playwright: {url}")
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(3) # Short sleep to let dynamic JS elements resolve safely
                
                # Fetch text content inside the HTML body tag
                body_text = page.locator("body").inner_text()
                
                # Turn it into a LangChain document object matching WebBaseLoader's format
                doc = Document(page_content=body_text, metadata={"source": url})
                documents.append(doc)
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
                
        browser.close()
    return documents


def index_website(urls=BETTERHELP_URLS, reset=True):
    """
    Scrape the URLs, chunk them, and upsert into Pinecone.
    reset=True clears the namespace first so removed/shortened
    pages don't leave stale chunks behind.
    Returns the number of chunks upserted.
    """
    #docs = WebBaseLoader(urls).load()
    docs = scrape_with_playwright(urls)

    if not docs:
        print("No documents were successfully extracted.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    index = get_pinecone_index()

    if reset:
        try:
            index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        except Exception:
            pass  # namespace doesn't exist yet on the very first run

    vector_store = PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
    )
    vector_store.add_documents(chunks, ids=make_chunk_ids(chunks), batch_size=100)

    return len(chunks)

def build_retriever(urls=BETTERHELP_URLS, k=4):
    """
    Connect to Pinecone. Only scrapes + embeds if the namespace is empty,
    so app restarts no longer re-index the whole website.
    """
    index = get_pinecone_index()

    if namespace_vector_count(index) == 0:
        index_website(urls, reset=False)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


examples = [
    {
        "input": "Who are you?",
        "output": "I'm your AI assistant. My name is Ruchi.I can help you with questions about Mood Swing Disorder. "
                  "I will match you to one of 31,000 licensed therapists based upon your location, preferences, and therapist availability.",
    },
    {
        "input": "I'm from Texas. What type of Therapy I can get?",
        "output": "We can provide you three kind of therapies. They are based on Individual, couples or teen therapy. "
                  "We start from fundamentals and choose therapy , step by step. "
                  "No prior appointment needed to begin.",
    },
    {
        "input": "How do you choose therapy for a patient?",
        "output": "We strive to connect you with a therapist who aligns with your needs and preferences."
                  "To make this process as smooth as possible, we use automated tools, including machine learning, to help find the best match based on relevant factors, such as fit, type of support, and long-term engagement."
                  "While our service is designed to enhance your experience and the best outcome, you always have the option to switch to a new therapist in your account settings or through our contact us page.",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai","{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
           "You are a helpful, friendly AI assitant built by Organization BetterHelp."
            "Your name is Ruchi."
            "You give answers related to mental health."
            "For other than health questions,you can say that you cannot provide information regarding that. You politely refuse them."
            "You remember the ongoing conversation and answer clearly, short and concisely. "
            "You will always give step by step process how a therapy can make their life better. "
            "the context, say you're not sure and suggest contacting BETTER Help directly — "
            "don't make things up. Match the tone of the examples & For More Queries, Reach BETTER HELP Team: +1 409-444-5555\n\n"
            "BETTER HELP website context:\n{context}",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def demo_memory():
    return InMemoryChatMessageHistory()

def demo_conversation(input_text, memory, retriever):
    llm = demo_chatbot()

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | CHAT_PROMPT
        | llm
    )

    chat_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "betterhelp-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory