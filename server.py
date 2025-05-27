from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import firebase_admin
from firebase_admin import credentials, auth, firestore
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize Firebase
cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize OpenAI and Tavily
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("Error: OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")
openai.api_key = openai_api_key
try:
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    print("ChatOpenAI initialized successfully")
except Exception as e:
    print(f"Error initializing ChatOpenAI: {e}")
    raise
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

@app.route('/')
def index():
    return jsonify({"message": "AI Agent Builder Backend"})

def verify_token(token):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

@app.route('/create_agent', methods=['POST'])
async def create_agent():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Unauthorized"}), 401
    token = auth_header.split(' ')[1]
    user_id = verify_token(token)
    if not user_id:
        return jsonify({"error": "Invalid token"}), 401

    data = request.get_json()
    if not data or 'name' not in data or 'prompt' not in data:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        doc_ref = db.collection('agents').document()
        doc_ref.set({
            'user_id': user_id,
            'name': data['name'],
            'prompt': data['prompt'],
            'tools': data.get('tools', []),
            'trainingData': data.get('trainingData', ''),
            'created_at': firestore.SERVER_TIMESTAMP
        })
        return jsonify({"message": "Agent created successfully"}), 200
    except Exception as e:
        print(f"Error creating agent: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
async def chat():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Unauthorized"}), 401
    token = auth_header.split(' ')[1]
    user_id = verify_token(token)
    if not user_id:
        return jsonify({"error": "Invalid token"}), 401

    data = request.get_json()
    if not data or 'message' not in data or 'agent_name' not in data:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Fetch agent configuration
        agent_docs = db.collection('agents').where('user_id', '==', user_id).where('name', '==', data['agent_name']).get()
        if not agent_docs:
            return jsonify({"error": "Agent not found"}), 404
        agent_data = agent_docs[0].to_dict()

        # Create agent
        system_message = agent_data['prompt']
        agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

        # Process message
        response = await agent_executor.ainvoke({"messages": [{"role": "user", "content": data['message']}]})
        assistant_response = response['messages'][-1]['content']

        # Save to Firestore
        db.collection('chat_history').add({
            'user_id': user_id,
            'agent_name': data['agent_name'],
            'message': data['message'],
            'response': assistant_response,
            'created_at': firestore.SERVER_TIMESTAMP
        })

        return jsonify({"response": assistant_response}), 200
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/widget/<user_id>/<agent_name>')
def widget(user_id, agent_name):
    anon_token = os.getenv('FIREBASE_ANON_TOKEN')
    if not anon_token:
        print("Error: FIREBASE_ANON_TOKEN not configured")
        return "Error: Anonymous token not configured", 500
    print(f"Rendering widget for user_id: {user_id}, agent_name: {agent_name}")
    return f"""<html>
<body>
    <h1>Chat with {agent_name}</h1>
    <div id="chat"></div>
    <input id="message" type="text" placeholder="Type your message...">
    <button onclick="sendMessage()">Send</button>
    <script>
        async function sendMessage() {{
            const message = document.getElementById('message').value;
            try {{
                const response = await fetch('https://agentbuilderbackend.onrender.com/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json', 'Authorization': 'Bearer {anon_token}' }},
                    body: JSON.stringify({{ message, user_id: '{user_id}', agent_name: '{agent_name}' }})
                }});
                const data = await response.json();
                const chat = document.getElementById('chat');
                const responseText = data.response ? data.response : 'Error';
                chat.innerHTML += `<p>User: ${message}</p><p>Agent: ${responseText}</p>`;
                document.getElementById('message').value = '';
            }} catch (error) {{
                console.error('Error:', error);
                alert('Failed to send message');
            }}
        }}
    </script>
</body>
</html>
"""

@app.route('/refresh-token', methods=['POST'])
async def refresh_token():
    try:
        import requests
        response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={os.getenv('FIREBASE_API_KEY')}",
            json={"returnSecureToken": true}
        )
        data = response.json()
        if data.get('idToken'):
            return jsonify({"idToken": data['idToken']}), 200
        return jsonify({"error": "Failed to generate token"}), 500
    except Exception as e:
        print(f"Error refreshing token: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)