# HotelsByDay Chatbot Backend

A FastAPI-based chatbot backend with confidence-based escalation to human agents, powered by Groq LLM and Chroma vector store.

## 🚀 Quick Start

### Option 1: Using the Activation Script (Recommended)
```bash
cd chatbot
./activate_venv.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📋 Prerequisites

- Python 3.8+
- pip
- Groq API key (set as environment variable)

## 🔧 Environment Setup

Create a `.env` file in the chatbot directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

## 🌐 API Endpoints

### Chat Endpoints
- `POST /chat` - Main chat endpoint with confidence scoring
- `POST /agent-chat` - Human agent response endpoint
- `GET /session/{session_id}/status` - Get session status
- `GET /agent/sessions` - List escalated sessions

### Admin Endpoints
- `POST /update-website` - Update knowledge base from website
- `POST /upload-pdf` - Upload and index PDF documents

## 🎯 Features

### Confidence-Based Escalation
- Automatic escalation when confidence score < 0.4
- User request escalation triggers
- Seamless handoff to human agents
- Session continuity for agents

### RAG (Retrieval-Augmented Generation)
- Website content indexing
- PDF document processing
- Context-aware responses
- Vector similarity search

### Session Management
- Persistent chat history
- Agent assignment tracking
- Escalation timestamps
- Confidence score tracking

## 🏗️ Project Structure

```
chatbot/
├── main.py              # FastAPI application
├── groq_client.py       # Groq LLM integration
├── vectorstore.py       # Chroma vector store
├── utils.py             # Utility functions
├── db.py                # SQLite database
├── scraper.py           # Website scraping
├── requirements.txt     # Python dependencies
├── .venv/               # Virtual environment
└── activate_venv.sh     # Activation script
```

## 🔍 Debugging

### VS Code Configuration
The project includes VS Code configuration files in `.vscode/`:
- `launch.json` - Debug configurations
- `tasks.json` - Build and development tasks
- `settings.json` - Workspace settings

### Debug Configurations
1. **Python: FastAPI Backend** - Debug the main server
2. **Python: Debug Chat Endpoint** - Focus on chat functionality
3. **Full Stack: Backend + Frontend** - Debug both simultaneously

### Tasks
- **pip: install** - Install Python dependencies
- **python: start backend** - Start FastAPI server
- **Start Full Stack** - Launch backend and frontend
- **Test Chat API** - Test the chat endpoint

## 🧪 Testing

### Test the Chat API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_session", "message": "Hello, how can you help me?"}'
```

### Test Escalation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_session", "message": "I want to speak to a human agent"}'
```

## 📊 Monitoring

### Logs
The server provides detailed logging:
- User messages and responses
- Confidence scores
- Escalation triggers
- Agent assignments

### Database
SQLite database stores:
- Chat summaries
- User contact information
- Agent assignments
- Escalation timestamps

## 🔧 Configuration

### Confidence Threshold
Edit `LOW_CONFIDENCE_THRESHOLD` in `main.py`:
```python
LOW_CONFIDENCE_THRESHOLD = 0.4  # Adjust as needed
```

### Model Configuration
Set in `.env` file:
```env
GROQ_MODEL=llama3-8b-8192  # or other Groq models
```

## 🚀 Deployment

### Production Setup
1. Use production WSGI server (Gunicorn)
2. Set up proper environment variables
3. Configure CORS for frontend domain
4. Set up SSL certificates

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Integration

### Frontend Integration
The backend is designed to work with the Next.js frontend:
- CORS configured for frontend domain
- Session management compatible
- Real-time escalation support

### External Services
- **Groq LLM** - For chat responses and confidence scoring
- **Chroma DB** - For vector storage and retrieval
- **Formspree** - For human agent notifications (configurable)

## 🐛 Troubleshooting

### Common Issues

1. **Virtual Environment Not Found**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Groq API Key Not Set**
   ```bash
   export GROQ_API_KEY=your_key_here
   ```

4. **Port Already in Use**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8001 --reload
   ```

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance

### Optimization Tips
- Use connection pooling for database
- Implement caching for frequent queries
- Optimize vector search parameters
- Monitor memory usage for large documents

### Monitoring
- Response times
- Memory usage
- Database performance
- Vector store efficiency

## 🔒 Security

### Best Practices
- Validate all inputs
- Use environment variables for secrets
- Implement rate limiting
- Sanitize user messages
- Use HTTPS in production

## 📝 License

This project is licensed under the MIT License. 