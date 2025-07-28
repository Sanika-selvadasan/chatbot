#!/usr/bin/env python3
"""
Test script for admin chat functionality
"""

import asyncio
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_admin_chat_flow():
    """Test the complete admin chat flow"""
    
    print("🧪 Testing Admin Chat Functionality")
    print("=" * 50)
    
    # 1. Create a chat session and escalate it
    print("\n1. Creating and escalating a chat session...")
    
    session_id = f"test_session_{int(time.time())}"
    
    # Send a message that should trigger escalation
    chat_response = requests.post(f"{BASE_URL}/chat", json={
        "session_id": session_id,
        "message": "I want to book a room for tomorrow at 2pm"
    })
    
    if chat_response.status_code == 200:
        chat_data = chat_response.json()
        print(f"✅ Chat response: {chat_data['escalated']}")
        print(f"✅ Agent ID: {chat_data.get('agent_id')}")
        
        if chat_data['escalated']:
            agent_id = chat_data['agent_id']
            
            # 2. Get escalated sessions
            print("\n2. Getting escalated sessions...")
            sessions_response = requests.get(f"{BASE_URL}/agent/sessions")
            
            if sessions_response.status_code == 200:
                sessions_data = sessions_response.json()
                print(f"✅ Found {len(sessions_data['escalated_sessions'])} escalated sessions")
                
                # 3. Take the session
                print(f"\n3. Taking session with agent {agent_id}...")
                take_response = requests.post(f"{BASE_URL}/agent/sessions/{agent_id}/take")
                
                if take_response.status_code == 200:
                    take_data = take_response.json()
                    print(f"✅ Session taken: {take_data['message']}")
                    
                    # 4. Get session history
                    print(f"\n4. Getting session history...")
                    history_response = requests.get(f"{BASE_URL}/session/{session_id}/history")
                    
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        print(f"✅ Session history loaded: {len(history_data['history'])} messages")
                        
                        # 5. Send agent message
                        print(f"\n5. Sending agent message...")
                        agent_message = "Hello! I'm a human agent here to help you with your booking request. How can I assist you today?"
                        
                        agent_response = requests.post(f"{BASE_URL}/agent-chat", json={
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "message": agent_message
                        })
                        
                        if agent_response.status_code == 200:
                            agent_data = agent_response.json()
                            print(f"✅ Agent message sent: {agent_data['message']}")
                            
                            # 6. Verify the message was added to history
                            print(f"\n6. Verifying message in history...")
                            updated_history = requests.get(f"{BASE_URL}/session/{session_id}/history")
                            
                            if updated_history.status_code == 200:
                                updated_data = updated_history.json()
                                print(f"✅ Updated history: {len(updated_data['history'])} messages")
                                
                                # Check if agent message is in history
                                agent_messages = [msg for msg in updated_data['history'] if msg.get('role') == 'agent']
                                if agent_messages:
                                    print(f"✅ Agent message found in history: {agent_messages[-1]['content']}")
                                else:
                                    print("❌ Agent message not found in history")
                            else:
                                print(f"❌ Failed to get updated history: {updated_history.status_code}")
                        else:
                            print(f"❌ Failed to send agent message: {agent_response.status_code}")
                    else:
                        print(f"❌ Failed to get session history: {history_response.status_code}")
                else:
                    print(f"❌ Failed to take session: {take_response.status_code}")
            else:
                print(f"❌ Failed to get escalated sessions: {sessions_response.status_code}")
        else:
            print("❌ Session was not escalated")
    else:
        print(f"❌ Failed to create chat session: {chat_response.status_code}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_admin_chat_flow() 