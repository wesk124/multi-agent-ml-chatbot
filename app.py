import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone


class SimpleEmbeddings:
    """Lightweight wrapper around OpenAI Embeddings API (no langchain needed)."""
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed_query(self, text: str) -> list:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


# ─── Agent Classes (copied from notebook for standalone deployment) ───────────


class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = (
            "You are a content moderation assistant. Your sole job is to determine "
            "whether a user's query contains ANY obnoxious, rude, hateful, insulting, "
            "or inappropriate language. "
            "If the query contains ANY insults, slurs, name-calling, profanity, or hostile tone "
            "— even if it ALSO contains a legitimate question — respond 'Yes'. "
            "A query like 'idiot! what is machine learning' is obnoxious because it contains 'idiot'. "
            "A query like 'explain this you moron' is obnoxious because it contains 'moron'. "
            "Only respond 'No' if the query is entirely polite and respectful. "
            "Respond with exactly one word: 'Yes' or 'No'. Do not provide any other output."
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        text = response.choices[0].message.content.strip().lower()
        return text.startswith("yes")

    def check_query(self, query):
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        return self.extract_action(response)


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings, namespace="ns2500") -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.namespace = namespace
        self.prompt = (
            "You are a relevance-checking assistant. Given a user query and a list of "
            "retrieved document snippets, determine if the documents are relevant to the "
            "query. Respond with exactly 'Relevant' or 'Irrelevant'. Do not provide any "
            "other output."
        )

    def query_vector_store(self, query, k=5, score_threshold=0.25):
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding, top_k=k, include_metadata=True,
            namespace=self.namespace
        )
        docs = []
        for match in results["matches"]:
            if match["score"] >= score_threshold:
                text = match["metadata"].get("text", "")
                docs.append({"text": text, "score": match["score"], "id": match["id"]})
        return docs

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response, query=None):
        text = response.choices[0].message.content.strip().lower()
        return "relevant" in text and "irrelevant" not in text


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a helpful assistant that answers questions based on the provided "
            "context documents. Use the documents to form your answer. If the documents "
            "do not contain enough information, say so honestly. Be concise and accurate."
        )

    def generate_response(self, query, docs, conv_history, k=5):
        context = "\n\n".join(
            [f"Document {i+1}: {doc['text']}" for i, doc in enumerate(docs[:k])]
        )
        messages = [{"role": "system", "content": self.prompt}]
        for msg in conv_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context documents:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Please answer the question based on the context documents above."
                ),
            }
        )
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a relevance evaluation assistant. Given a user query and a set of "
            "retrieved documents, determine if the documents are related to the topic "
            "of the query and could help answer it. "
            "Respond 'Relevant' if the documents discuss the same general topic as the query "
            "or contain information that would be useful for answering it. "
            "Respond 'Irrelevant' ONLY if the documents are about a completely different, "
            "unrelated subject (e.g., query is about cooking but documents are about machine learning). "
            "Respond with exactly one word: 'Relevant' or 'Irrelevant'. No other output."
        )

    def get_relevance(self, query, documents) -> str:
        docs_text = "\n\n".join(
            [f"Document {i+1}: {doc['text']}" for i, doc in enumerate(documents)]
        )
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": (
                        f"User query: {query}\n\n"
                        f"Retrieved documents:\n{docs_text}\n\n"
                        "Are the documents relevant to the query?"
                    ),
                },
            ],
            temperature=0,
        )
        text = response.choices[0].message.content.strip().lower()
        if "irrelevant" in text:
            return "Irrelevant"
        return "Relevant"


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name, namespace="ns2500") -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name
        self.namespace = namespace

        self.client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        self.embeddings = SimpleEmbeddings(self.client)

        self.conversation_history = []
        self.setup_sub_agents()

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.query_agent = Query_Agent(self.pinecone_index, self.client, self.embeddings, self.namespace)
        self.answering_agent = Answering_Agent(self.client)
        self.relevant_docs_agent = Relevant_Documents_Agent(self.client)

    def _is_small_talk(self, query) -> bool:
        """Detect greetings / small talk using LLM."""
        resp = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": (
                    "You are a classifier. Determine if the user's message is a greeting "
                    "or casual small talk (e.g. 'hello', 'hi', 'good morning', 'how are you', "
                    "'hey there'). Respond with exactly 'Yes' or 'No'."
                )},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip().lower().startswith("yes")

    def handle_query(self, query):
        """Process a single user query through the agent pipeline.
        Returns (response, agent_used, trace) where trace is a list of step dicts.
        """
        trace = []

        # Step 1: Obnoxious Agent
        is_obnoxious = self.obnoxious_agent.check_query(query)
        trace.append({
            "step": 1,
            "agent": "Obnoxious_Agent",
            "input": query,
            "output": "Yes (blocked)" if is_obnoxious else "No (safe)",
        })
        if is_obnoxious:
            msg = ("I'm sorry, but I can't respond to that kind of language. "
                   "Please rephrase your query respectfully.")
            return msg, "Obnoxious_Agent", trace

        # Step 2: Query Agent (Pinecone)
        docs = self.query_agent.query_vector_store(query)
        trace.append({
            "step": 2,
            "agent": "Query_Agent",
            "input": query,
            "output": f"Retrieved {len(docs)} documents (above score threshold)",
        })

        # Early exit: no docs passed the score threshold
        if not docs:
            # Check if it's small talk / greeting before refusing
            if self._is_small_talk(query):
                trace.append({
                    "step": 3,
                    "agent": "Head_Agent",
                    "input": query,
                    "output": "Detected as greeting/small talk",
                })
                msg = ("Hello! I'm a Machine Learning chatbot. I can answer questions "
                       "about topics like regression, neural networks, decision trees, "
                       "and more. How can I help you?")
                return msg, "Head_Agent", trace
            trace.append({
                "step": 3,
                "agent": "Query_Agent",
                "input": query,
                "output": "No documents above similarity threshold — query is off-topic",
            })
            msg = ("I'm sorry, but I don't have information related to your query "
                   "in my knowledge base. I can only answer questions about the "
                   "topics covered in my documents.")
            return msg, "Query_Agent", trace

        # Step 3: Relevant Documents Agent
        relevance = self.relevant_docs_agent.get_relevance(query, docs)
        trace.append({
            "step": 3,
            "agent": "Relevant_Documents_Agent",
            "input": f"{query} + {len(docs)} docs",
            "output": relevance,
        })
        if relevance == "Irrelevant":
            msg = ("I'm sorry, but I don't have information related to your query "
                   "in my knowledge base. I can only answer questions about the "
                   "topics covered in my documents.")
            return msg, "Relevant_Documents_Agent", trace

        # Step 4: Answering Agent
        answer = self.answering_agent.generate_response(
            query, docs, self.conversation_history
        )
        trace.append({
            "step": 4,
            "agent": "Answering_Agent",
            "input": f"{query} + {len(docs)} docs",
            "output": answer[:100] + "..." if len(answer) > 100 else answer,
        })

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer, "Answering_Agent", trace


# ─── Streamlit App ────────────────────────────────────────────────────────────


st.set_page_config(page_title="Multi-Agent ML Chatbot", page_icon="🤖", layout="centered")

# ── Password Gate ─────────────────────────────────────────────────────────────
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "eep596")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Multi-Agent ML Chatbot")
    st.markdown("Please enter the password to access the chatbot.")
    pwd = st.text_input("Password", type="password", placeholder="Enter password...")
    if st.button("Login", use_container_width=True):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

# ── User is authenticated beyond this point ───────────────────────────────────
st.title("🤖 Multi-Agent ML Chatbot")
st.caption("Powered by GPT-4.1-nano | Pinecone RAG | Multi-Agent Pipeline")

# ── Helper: read API keys from Streamlit secrets or environment ───────────────
def _get_secret(key, default=""):
    """Try st.secrets first, then env vars, then default."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

# ── Auto-initialize agent from secrets on first run ──────────────────────────
if "head_agent" not in st.session_state:
    _ok = _get_secret("OPENAI_API_KEY")
    _pk = _get_secret("PINECONE_API_KEY")
    _idx = _get_secret("PINECONE_INDEX_NAME", "machine-learning-textbook")
    _ns = _get_secret("PINECONE_NAMESPACE", "ns2500")
    if _ok and _pk:
        with st.spinner("Auto-initializing agent from saved secrets..."):
            try:
                st.session_state.head_agent = Head_Agent(_ok, _pk, _idx, _ns)
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Auto-init failed: {e}")

# ── Sidebar: API key configuration ────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Show whether default keys are loaded from secrets
    _has_defaults = bool(_get_secret("OPENAI_API_KEY") and _get_secret("PINECONE_API_KEY"))
    if _has_defaults:
        st.info("Default API keys loaded from server secrets.")

    openai_key = st.text_input(
        "OpenAI API Key",
        value="",
        type="password",
        placeholder="Using default key" if _has_defaults else "Enter your key...",
        help="Leave blank to use the default key, or enter your own.",
    )
    pinecone_key = st.text_input(
        "Pinecone API Key",
        value="",
        type="password",
        placeholder="Using default key" if _has_defaults else "Enter your key...",
        help="Leave blank to use the default key, or enter your own.",
    )
    pinecone_index = st.text_input(
        "Pinecone Index Name",
        value=_get_secret("PINECONE_INDEX_NAME", "machine-learning-textbook"),
        help="Name of your Pinecone index",
    )
    pinecone_namespace = st.text_input(
        "Pinecone Namespace",
        value=_get_secret("PINECONE_NAMESPACE", "ns2500"),
        help="Namespace within the Pinecone index (e.g., ns500, ns1000, ns2500)",
    )

    # Use user-provided keys if filled, otherwise fall back to secrets
    openai_key = openai_key or _get_secret("OPENAI_API_KEY")
    pinecone_key = pinecone_key or _get_secret("PINECONE_API_KEY")

    st.divider()

    # Initialize / re-initialize button
    init_ready = all([openai_key, pinecone_key, pinecone_index, pinecone_namespace])
    if st.button("Initialize Agent", disabled=not init_ready, use_container_width=True):
        with st.spinner("Connecting to OpenAI & Pinecone..."):
            try:
                st.session_state.head_agent = Head_Agent(
                    openai_key, pinecone_key, pinecone_index, pinecone_namespace
                )
                st.session_state.messages = []
                st.success("Agent initialized successfully!")
            except Exception as e:
                st.error(f"Initialization failed: {e}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if "head_agent" in st.session_state:
            st.session_state.head_agent.conversation_history = []
        st.rerun()

    # Status indicator
    if "head_agent" in st.session_state:
        st.sidebar.success("Agent: Online")
    else:
        st.sidebar.warning("Agent: Not initialized")

    # ── Agent Testing Panel ───────────────────────────────────────────────────
    st.divider()
    st.header("Agent Testing")

    test_agent = st.selectbox("Select agent to test", [
        "Obnoxious Agent",
        "Relevant Docs Agent",
    ])

    test_query = st.text_input("Test query", placeholder="Type a test query...")

    if st.button("Run Test", use_container_width=True):
        if "head_agent" not in st.session_state:
            st.error("Initialize the agent first!")
        elif not test_query:
            st.warning("Enter a test query.")
        else:
            ha = st.session_state.head_agent
            with st.spinner("Testing..."):
                if test_agent == "Obnoxious Agent":
                    result = ha.obnoxious_agent.check_query(test_query)
                    if result:
                        st.error(f"**Obnoxious: Yes** — Query would be refused.")
                    else:
                        st.success(f"**Obnoxious: No** — Query is safe, will proceed.")

                elif test_agent == "Relevant Docs Agent":
                    docs = ha.query_agent.query_vector_store(test_query)
                    relevance = ha.relevant_docs_agent.get_relevance(test_query, docs)
                    if relevance == "Relevant":
                        st.success(f"**Relevant** — Found {len(docs)} matching docs.")
                    else:
                        st.warning(f"**Irrelevant** — No useful docs found.")
                    with st.expander("Retrieved documents"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Doc {i+1}** (score: {doc['score']:.4f})")
                            st.text(doc["text"][:300] + "...")

    # ── Batch test: Obnoxious Agent ───────────────────────────────────────────
    if st.checkbox("Run batch test (Obnoxious Agent)"):
        if "head_agent" not in st.session_state:
            st.error("Initialize the agent first!")
        else:
            batch_cases = [
                ("Explain ML, you idiot", True),
                ("You're so dumb, what is regression?", True),
                ("Shut up and tell me about neural networks", True),
                ("This is the stupidest AI ever", True),
                ("What is gradient descent?", False),
                ("Can you explain logistic regression?", False),
                ("Hello, how are you?", False),
                ("Tell me about decision trees", False),
            ]
            ha = st.session_state.head_agent
            passed = 0
            rows = []
            with st.spinner("Running 8 test cases..."):
                for query, expected in batch_cases:
                    result = ha.obnoxious_agent.check_query(query)
                    ok = result == expected
                    if ok:
                        passed += 1
                    rows.append({
                        "Query": query,
                        "Expected": "Obnoxious" if expected else "Safe",
                        "Got": "Obnoxious" if result else "Safe",
                        "Result": "PASS" if ok else "FAIL",
                    })
            st.dataframe(rows, use_container_width=True)
            if passed == len(batch_cases):
                st.success(f"All {passed}/{len(batch_cases)} tests passed!")
            else:
                st.warning(f"{passed}/{len(batch_cases)} tests passed.")

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "agent" in msg:
            st.caption(f"Handled by: `{msg['agent']}`")
            if msg.get("trace"):
                with st.expander("View agent pipeline trace"):
                    for step in msg["trace"]:
                        icon = {
                            "Obnoxious_Agent": "🛡️",
                            "Query_Agent": "🔍",
                            "Relevant_Documents_Agent": "📄",
                            "Answering_Agent": "💡",
                        }.get(step["agent"], "⚙️")
                        st.markdown(
                            f"**Step {step['step']}** {icon} `{step['agent']}`\n\n"
                            f"- **Input:** {step['input'][:150]}\n"
                            f"- **Output:** {step['output'][:150]}"
                        )
                        st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me about Machine Learning..."):
    # Guard: agent must be initialized
    if "head_agent" not in st.session_state:
        st.error("Please configure your API keys and click **Initialize Agent** in the sidebar first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from multi-agent pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, agent_used, trace = st.session_state.head_agent.handle_query(prompt)
                except Exception as e:
                    response = f"An error occurred: {e}"
                    agent_used = "Error"
                    trace = []

            st.markdown(response)
            st.caption(f"Handled by: `{agent_used}`")

            # Show agent pipeline trace
            if trace:
                with st.expander("View agent pipeline trace"):
                    for step in trace:
                        icon = {
                            "Obnoxious_Agent": "🛡️",
                            "Query_Agent": "🔍",
                            "Relevant_Documents_Agent": "📄",
                            "Answering_Agent": "💡",
                        }.get(step["agent"], "⚙️")
                        st.markdown(
                            f"**Step {step['step']}** {icon} `{step['agent']}`\n\n"
                            f"- **Input:** {step['input'][:150]}\n"
                            f"- **Output:** {step['output'][:150]}"
                        )
                        st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "agent": agent_used, "trace": trace}
        )
