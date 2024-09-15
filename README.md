
# **AISA (AI Support Assistant)** – Devtitans  
## Project Overview  

AISA is a next-generation AI-powered customer support platform, built to revolutionize how companies engage with their customers. It integrates cutting-edge technologies, such as the Retrieval-Augmented Generation (RAG) model, generative AI using Llama 3.1, and powerful multi-modal data handling with Gemini 1.5 Flash API for a seamless experience across text, voice, and image inputs. The platform is optimized for the Indian context and offers scalable solutions to serve global needs.

The platform consists of two key components:
- **Customer-Service-App (Next.js)**: A user-friendly interface where customers interact with the AI for support.
- **Streamlit Dashboard**: A comprehensive dashboard for customer support agents to monitor and assist customers, using data retrieval from the RAG pipeline and other key metrics.

# Repository 2: **Streamlit Dashboard**  
This repository contains the Streamlit-based dashboard that allows customer service agents to monitor queries and assist customers when needed. The dashboard connects to the RAG pipeline and retrieves documents stored in Pinecone, providing a smooth experience for the agents.

### 1. **Features**
- **Real-time Monitoring**: Agents can monitor customer queries in real-time and see responses generated by the AI.
- **Query Escalation**: The dashboard highlights customer queries that couldn’t be solved by the AI, allowing human agents to intervene.
- **Voice & Text Integration**: Agents can interact with the system via text or voice, and they can see queries submitted in both formats.
- **Sentiment Analysis**: Real-time sentiment analysis helps agents understand the tone and satisfaction level of the customer.

### 2. **Technologies Used**
- **Streamlit**: A Python framework for creating interactive dashboards.
- **Pinecone**: For retrieving vectorized documents based on embeddings produced by Mistral.
- **Llama 3.1**: For generating responses when needed.
- **Gemini 1.5 Flash API**: For voice and image queries processed via the dashboard.
- **Twilio**: For voice interaction and communication between customers and agents.
  
### 3. **Installation**
To get started with the **streamlit_dashboard**, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Devtitans/streamlit_dashboard.git

# Navigate to the project directory
cd streamlit_dashboard

# Install dependencies
pip install -r requirements.txt

# Start the dashboard
streamlit run dashboard.py
```

You will need to provide API keys for **Gemini 1.5 Flash**, **Pinecone**, and **Twilio** in a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
TWILIO_API_KEY=your_twilio_api_key
```

### 4. **File Structure**
Here’s a breakdown of the important files in this repository:

```bash
streamlit_dashboard/
├── dashboard.py  # The main file that starts the Streamlit dashboard
├── components/
│   ├── agent_view.py  # Displays real-time agent interaction UI
│   └── query_table.py  # Displays a table of recent customer queries
├── services/
│   ├── pinecone_service.py  # Handles Pinecone document retrieval
│   └── gemini_service.py  # Processes voice and image inputs using Gemini
└── requirements.txt  # List of dependencies
```

### 5. **Usage**
Once the dashboard is running, customer support agents can:
- View incoming queries in real-time.
- Respond to complex queries that AI couldn’t handle.
- Use sentiment analysis to gauge customer satisfaction.
- Listen to voice queries and view image-based queries converted by Gemini 1.5 Flash API.

### 6. **Challenges Faced**
The major challenge here was managing real-time data retrieval for large sets of customer queries. The integration of **Pinecone** helped us scale the document retrieval process, but tweaking the Mistral embeddings and ensuring fast, reliable responses required a lot of fine-tuning. We also had to ensure that the dashboard handled queries from multiple sources (voice, text, image) without creating significant delays or errors.

---

# **Conclusion**  

The **AISA** platform, developed by **Devtitans**, aims to solve one of the most pressing issues in customer service today – providing fast, reliable, and accessible support across multiple channels. By leveraging **Next.js**, **Streamlit**, **Pinecone**, **Gemini 1.5 Flash API**, and **Mistral**, we've built a comprehensive system that can understand and process text, voice, and images, while providing accurate, contextually-relevant responses.

We encountered challenges, especially with multi-modal input processing and document retrieval, but by pivoting to more robust solutions like **Gemini 1.5** and **Pinecone**, we were able to overcome these hurdles. The result is a platform that is not only scalable but also adaptable to a global audience, with real-time multi-language support and robust AI-powered customer interactions.

