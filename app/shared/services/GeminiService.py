import os
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Optional Google / Gemini / Langchain specific
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None
    HumanMessage = None
    LANGCHAIN_AVAILABLE = False

# Optional Vector Store - Standard LangChain integration
try:
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_google_vertexai import VertexAIEmbeddings
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SupabaseVectorStore = None
    create_client = None
    SUPABASE_AVAILABLE = False

# Import tools and services
from app.shared.utils.toolhub import get_all_tools
from app.shared.utils.conversation_state import State, ConversationState

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeminiService:
    MODEL_NAME = "gemini-2.5-flash"  # LLM for generation
    TEMPERATURE = 0.3  # Lower temperature for factual accuracy
    TIMEOUT = 60
    MAX_RETRIES = 3
    ERROR_MESSAGE = "Sorry, the Hermes LLM service encountered an error."

    # Using text-embedding-004 (Gemini API compatible)
    # Note: text-embedding-005 is Vertex AI only, not available via Gemini API
    EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # Gemini API embedding model
    EMBEDDING_DIMENSIONS = 768  # Dimension for text-embedding-004
    TEXT_SPLITTER_CHUNK_SIZE = 1000  # Larger chunks for better context
    TEXT_SPLITTER_CHUNK_OVERLAP = 200  # More overlap for better context retention
    RAG_TOP_K = 5  # Reduced from 30 for better quality
    RAG_SIMILARITY_THRESHOLD = 0.65  # Raised to filter out low-quality matches

    def __init__(
        self,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        conversation_db_path: str = "conversations.db"
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. "
                "Install with: pip install langchain-google-genai langchain-google-vertexai langchain-core"
            )

        vertex_project = vertex_project or os.environ.get("GOOGLE_PROJECT_ID")
        vertex_location = vertex_location or os.environ.get("GOOGLE_PROJECT_LOCATION")

        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        # Initialize LLM
        base_model = ChatGoogleGenerativeAI(
            model=self.MODEL_NAME,
            temperature=self.TEMPERATURE,
            max_retries=self.MAX_RETRIES,
            timeout=self.TIMEOUT,
        )

        # Attach tools
        tools = get_all_tools()
        self.model = base_model.bind_tools(tools)
        logging.info("Initialized Gemini model with %d tools attached", len(tools))

        # Base prompts for different personas - load from markdown files
        self.base_prompts = self._load_agent_prompts()
        logging.info("Loaded base prompts for agents: %s", list(self.base_prompts.keys()))

        # Initialize Gemini embeddings (768 dimensions)
        try:
            # Configure the Gemini API
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            
            # Initialize the Google Generative AI embeddings model (uses API key, not Vertex AI)
            # This avoids needing Vertex AI authentication/permissions
            self.embeddings_model = GoogleGenerativeAIEmbeddings(
                model=self.EMBEDDING_MODEL_NAME,
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
            logging.info(
                "Initialized Gemini Embeddings with model: %s (%dD)",
                self.EMBEDDING_MODEL_NAME,
                self.EMBEDDING_DIMENSIONS
            )
        except Exception as e:
            logging.error("Failed to initialize Gemini Embeddings: %s", e)
            logging.error(traceback.format_exc())
            raise

        # Initialize Supabase vector store
        if not SUPABASE_AVAILABLE:
            logging.warning(
                "Supabase dependencies not available. "
                "Install with: pip install langchain-community supabase"
            )
            self.vector_store = None
        else:
            try:
                supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("SUPABASE_PROJECT_URL")
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                if not supabase_url or not supabase_key:
                    raise ValueError(
                        "SUPABASE_URL (or SUPABASE_PROJECT_URL) and SUPABASE_SERVICE_ROLE_KEY must be set."
                    )
                supabase_client = create_client(supabase_url, supabase_key)
                self.vector_store = SupabaseVectorStore(
                    client=supabase_client,
                    embedding=self.embeddings_model,
                    table_name="hermes_vectors"
                )
                logging.info("Initialized Supabase vector store using LangChain")
            except Exception as e:
                logging.error("Failed to initialize Supabase vector store: %s", e)
                logging.error(traceback.format_exc())
                raise

        # Initialize conversation state
        self.conversation_state = ConversationState(db_path=conversation_db_path)

        vector_store_status = "Supabase" if self.vector_store else "None (dependencies missing)"
        logging.info(
            "GeminiService initialized. LLM: %s, Embeddings: %s (%dD), Vector Store: %s",
            self.MODEL_NAME,
            self.EMBEDDING_MODEL_NAME,
            self.EMBEDDING_DIMENSIONS,
            vector_store_status
        )

    def _load_agent_prompts(self) -> Dict[str, str]:
        """
        Load agent prompts from markdown files in docs/AgentPrompts/.
        Returns a dictionary mapping agent names (lowercase) to their prompt content.
        """
        prompts = {}
        
        # Get the project root directory (3 levels up from this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        prompts_dir = os.path.join(project_root, "docs", "AgentPrompts")
        
        if not os.path.exists(prompts_dir):
            logging.warning("Agent prompts directory not found: %s", prompts_dir)
            return prompts
        
        # Load all .md files from the prompts directory
        try:
            for filename in os.listdir(prompts_dir):
                if filename.endswith(".md"):
                    agent_name = filename[:-3].lower()  # Remove .md extension and lowercase
                    filepath = os.path.join(prompts_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            prompts[agent_name] = content
                            logging.info(
                                "Loaded prompt for agent '%s' from %s (%d chars)",
                                agent_name, filename, len(content)
                            )
                    except Exception as e:
                        logging.error("Failed to load prompt from %s: %s", filepath, e)
        except Exception as e:
            logging.error("Error reading prompts directory %s: %s", prompts_dir, e)
        
        return prompts

    def _direct_similarity_search(self, query: str, k: int = 5, threshold: float = 0.7):
        """
        Direct Supabase vector similarity search using RPC.
        Bypasses broken LangChain wrapper.
        
        Args:
            query: Search query
            k: Number of results
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Call Supabase RPC function directly (note: _client with underscore)
            # The function signature is: match_documents(filter, match_count, query_embedding)
            response = self.vector_store._client.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_count': k * 2,  # Get more results to filter by threshold
                    'filter': {}  # Empty filter for now
                }
            ).execute()
            
            if not response.data:
                logging.warning("No data returned from match_documents RPC")
                return []
            
            # Log all similarity scores before filtering
            all_scores = [row.get('similarity', 0.0) for row in response.data]
            if all_scores:
                logging.info(
                    f"Raw similarity scores from DB: "
                    f"count={len(all_scores)}, "
                    f"max={max(all_scores):.3f}, "
                    f"min={min(all_scores):.3f}, "
                    f"avg={sum(all_scores)/len(all_scores):.3f}"
                )
            
            # Convert to LangChain Document format and filter by threshold
            from langchain_core.documents import Document
            results = []
            for row in response.data:
                similarity = row.get('similarity', 0.0)
                
                # Filter by threshold
                if similarity >= threshold:
                    doc = Document(
                        page_content=row.get('content', ''),
                        metadata=row.get('metadata', {})
                    )
                    results.append((doc, similarity))
            
            logging.info(f"After threshold filter: {len(results)}/{len(response.data)} chunks passed (threshold={threshold})")
            
            # Sort by similarity (descending) and limit to k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
            
        except Exception as e:
            logging.error(f"Direct similarity search failed: {e}")
            logging.error(traceback.format_exc())
            return []

    def generate_gemini_response(self, prompt: str, persona: str = 'hermes') -> str:
        base_prompt = self.base_prompts.get(persona, self.base_prompts['hermes'])
        prompt_prefix = f"{base_prompt}\n\n" if base_prompt else ""
        full_prompt = f"{prompt_prefix}User Input: {prompt}" if base_prompt else prompt
        try:
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            
            # Handle tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                logging.info(f"Model requested {len(message.tool_calls)} tool call(s)")
                tool_results = []
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    # Find and execute the tool
                    from app.shared.utils.toolhub import get_tool_by_name
                    tool = get_tool_by_name(tool_name)
                    if tool:
                        try:
                            result = tool._run(**tool_args)
                            tool_results.append(f"{tool_name}: {result}")
                            logging.info(f"Tool {tool_name} result: {result}")
                        except Exception as e:
                            error_msg = f"Error executing {tool_name}: {str(e)}"
                            tool_results.append(error_msg)
                            logging.error(error_msg)
                    else:
                        tool_results.append(f"Tool {tool_name} not found")
                
                # Return tool results formatted
                return "\n".join(tool_results)
            
            # Handle regular text response
            response = message.content.strip()
            if "error" in response.lower() or not response:
                logging.warning("Gemini returned error or empty: '%s'", response)
                return self.ERROR_MESSAGE + " Check safety filters or prompt."
            logging.info("Generated response successfully with persona: %s", persona)
            return response
        except Exception as e:
            logging.error("Error generating response: %s", e)
            logging.error(traceback.format_exc())
            return self.ERROR_MESSAGE + " Rate limiting or internal error. Try again in 30s."

    def _format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '').strip()
            if content:
                formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _enrich_query_for_vector_search(self, user_query: str, conversation_history: str = "") -> str:
        """
        Enrich a user query using Gemini to make it more detailed and suitable for vector search.
        Expands vague queries with context, synonyms, and related terms.
        
        Args:
            user_query: The original user query
            conversation_history: Optional conversation context
            
        Returns:
            Enriched query string optimized for vector similarity search
        """
        enrichment_prompt = f"""You are a query expansion expert. Your task is to
enrich the following user query to make it more suitable for semantic vector
search in a document database.

Rules:
1. Expand vague queries with context, synonyms, and related terms
2. Preserve the original intent and meaning
3. Add technical terms, domain-specific vocabulary, and common variations
4. Keep the enriched query concise (2-4 sentences max) (Under 300 chars)
5. Focus on searchable keywords and concepts, not conversational fluff
6. If the query is already detailed, return it with minor enhancements only

Conversation Context: {conversation_history}

Original Query: {user_query}

Enriched Query:"""

        try:
            # Use a simpler model call without tools for query enrichment
            base_model = ChatGoogleGenerativeAI(
                model=self.MODEL_NAME,
                temperature=0.3,  # Lower temp for focused enrichment
                max_retries=2,
                timeout=10,  # Shorter timeout for speed
            )
            
            message = base_model.invoke(
                [HumanMessage(content=enrichment_prompt)]
            )
            # Validate enrichment - check if it adds meaningful value
            enriched_query = message.content.strip()
            
            return enriched_query
            
        except Exception as e:
            logging.error(f"Error enriching query: {e}")
            logging.info("Falling back to original query")
            return user_query

    def generate_gemini_response_with_rag(
        self,
        prompt: str,
        user_id: str,
        persona: str = 'hermes',
        min_chunk_length: int = 20
    ) -> str:
        """
        Optimized RAG generation using Gemini embeddings and top-k.
        """

        # Retrieve or initialize conversation state
        state = self.conversation_state.get_state(user_id)
        if not state or 'conversation' not in state.data:
            state = State(data={
                'conversation': [],
                'metadata': {
                    'created_at': datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat(),
                    'message_count': 0
                }
            })

        # Add user message
        state.data['conversation'].append({
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.utcnow().isoformat()
        })
        state.data['metadata']['last_updated'] = datetime.utcnow().isoformat()
        state.data['metadata']['message_count'] += 1

        # Vector store fallback
        if self.vector_store is None:
            logging.info("Vector store not available. Using standard generation.")
            return self.generate_gemini_response(prompt, persona)

        # Retrieve top relevant chunks using DIRECT RPC call
        try:
            # Add entity context to improve matching
            contextualized_query = f"Edwin Kassier {prompt}"
            logging.info(f"Searching vector store with query: '{contextualized_query}'")
            
            # Use direct RPC call instead of broken LangChain wrapper
            relevant_chunks = self._direct_similarity_search(
                query=contextualized_query,
                k=self.RAG_TOP_K,
                threshold=self.RAG_SIMILARITY_THRESHOLD
            )
            
            if not relevant_chunks:
                logging.warning(
                    f"No chunks above similarity threshold {self.RAG_SIMILARITY_THRESHOLD}"
                )
                return self.generate_gemini_response(prompt, persona)
            
            # Log similarity scores for debugging
            scores = [score for _, score in relevant_chunks]
            logging.info(
                f"Retrieved {len(relevant_chunks)} high-quality chunks - "
                f"Scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}"
            )
            
            # Log top chunks for debugging
            for idx, (doc, score) in enumerate(relevant_chunks[:3]):
                preview = doc.page_content[:100].replace('\n', ' ')
                logging.info(f"  Chunk {idx+1}: score={score:.3f}, preview='{preview}...'")
            
            # Extract documents (already sorted by score)
            docs = [doc for doc, score in relevant_chunks]
            
            # Filter by minimum length
            docs = [doc for doc in docs if len(doc.page_content.strip()) >= min_chunk_length]
            
            if not docs:
                logging.info("No chunks passed length filter. Using standard generation.")
                return self.generate_gemini_response(prompt, persona)

        except Exception as e:
            logging.error("Error in vector search: %s", e)
            logging.error(traceback.format_exc())
            logging.info("Falling back to standard generation.")
            return self.generate_gemini_response(prompt, persona)

        # Conversation history
        conversation_history = self._format_conversation_history(
            state.data['conversation'][-10:]
        )

        # Base prompt
        base_prompt = self.base_prompts.get(
            persona, self.base_prompts['hermes']
        )
        prompt_prefix = f"{base_prompt}\n\n" if base_prompt else ""

        # Merge chunks
        context_str = "\n\n".join([doc.page_content for doc in docs])
        logging.info(f"Using {len(docs)} chunks ({len(context_str)} chars) for context")

        # Construct strict RAG prompt
        full_prompt = (
            f"{prompt_prefix}"
            f"Current UTC time: {datetime.utcnow().isoformat()}\n\n"
            f"=== CONVERSATION HISTORY ===\n{conversation_history}\n\n"
            f"=== VERIFIED CONTEXT (YOUR ONLY SOURCE OF TRUTH) ===\n{context_str}\n\n"
            f"=== USER'S MESSAGE ===\nUser: {prompt}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Answer ONLY using information explicitly stated in the VERIFIED CONTEXT above\n"
            f"2. If the VERIFIED CONTEXT doesn't contain the answer, respond: 'I don't have that information'\n"
            f"3. DO NOT use your general knowledge, training data, or make assumptions\n"
            f"4. DO NOT invent, extrapolate, or fill in missing details\n"
            f"5. When facts conflict, prefer the most recent information in the context\n\n"
            f"Your response:"
        )

        # Invoke LLM
        try:
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            response = message.content.strip()

            if not response or "error" in response.lower():
                logging.warning("RAG response error or empty: '%s'", response)
                return self.ERROR_MESSAGE + " RAG: Check rate limits or context."

            # Add assistant response to conversation state
            state.data['conversation'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.utcnow().isoformat()
            })
            state.data['metadata']['last_updated'] = datetime.utcnow().isoformat()

            # Save updated state
            self.conversation_state.save_state(user_id, state)
            logging.info("Generated RAG response successfully")
            return response

        except Exception as e:
            logging.error("Error generating RAG response: %s", e)
            logging.error(traceback.format_exc())
            return self.ERROR_MESSAGE + " RAG: Generation error."
