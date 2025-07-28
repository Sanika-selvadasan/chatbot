#!/usr/bin/env python3
"""
Test script for escalation persistence and message routing
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_escalation_persistence():
    """Test that escalation status persists and messages are routed correctly"""
    
    print("🧪 Testing Escalation Persistence and Message Routing")
    print("=" * 60)
    
    # 1. Create a session and escalate it
    print("\n1. Creating and escalating a session...")
    session_id = f"test_session_{int(time.time())}"
    
    chat_response = requests.post(f"{BASE_URL}/chat", json={
        "session_id": session_id,
        "message": "I want to book a room for tomorrow at 2pm"
    })
    
    if chat_response.status_code == 200:
        chat_data = chat_response.json()
        print(f"✅ Session created and escalated: {chat_data['escalated']}")
        print(f"✅ Agent ID: {chat_data.get('agent_id')}")
        
        if chat_data['escalated']:
            agent_id = chat_data['agent_id']
            
            # 2. Check escalated sessions
            print(f"\n2. Checking escalated sessions...")
            sessions_response = requests.get(f"{BASE_URL}/agent/sessions")
            
            if sessions_response.status_code == 200:
                sessions_data = sessions_response.json()
                print(f"✅ Found {len(sessions_data['escalated_sessions'])} escalated sessions")
                
                # 3. Take the session
                print(f"\n3. Taking session...")
                take_response = requests.post(f"{BASE_URL}/agent/sessions/{agent_id}/take")
                
                if take_response.status_code == 200:
                    take_data = take_response.json()
                    print(f"✅ Session taken successfully: {take_data['message']}")
                    
                    # 4. Send agent message
                    print(f"\n4. Sending agent message...")
                    agent_message = "Hello! I'm a human agent here to help you with your booking request."
                    
                    agent_response = requests.post(f"{BASE_URL}/agent-chat", json={
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "message": agent_message
                    })
                    
                    if agent_response.status_code == 200:
                        agent_data = agent_response.json()
                        print(f"✅ Agent message sent: {agent_data['message']}")
                        
                        # 5. Simulate user reply (this should be routed to admin)
                        print(f"\n5. Simulating user reply...")
                        user_reply = "Thank you! I need a room for 4 hours starting at 2pm tomorrow."
                        
                        # This should be routed to the admin, not processed by bot
                        user_response = requests.post(f"{BASE_URL}/chat", json={
                            "session_id": session_id,
                            "message": user_reply
                        })
                        
                        if user_response.status_code == 200:
                            user_data = user_response.json()
                            print(f"✅ User reply processed: {user_data['reply']}")
                            
                            # Check if the response indicates escalation (should not escalate again)
                            if "human agent" in user_data['reply'].lower():
                                print("✅ User message properly routed to human agent")
                            else:
                                print("❌ User message not properly routed to human agent")
                        
                        # 6. Check session history
                        print(f"\n6. Checking session history...")
                        history_response = requests.get(f"{BASE_URL}/session/{session_id}/history")
                        
                        if history_response.status_code == 200:
                            history_data = history_response.json()
                            print(f"✅ Session history loaded: {len(history_data['history'])} messages")
                            
                            # Check for both agent and user messages
                            agent_messages = [msg for msg in history_data['history'] if msg.get('role') == 'agent']
                            user_messages = [msg for msg in history_data['history'] if msg.get('role') == 'user']
                            
                            print(f"✅ Agent messages: {len(agent_messages)}")
                            print(f"✅ User messages: {len(user_messages)}")
                            
                            if agent_messages and user_messages:
                                print("✅ Both agent and user messages found in history")
                            else:
                                print("❌ Missing messages in history")
                        else:
                            print(f"❌ Failed to get session history: {history_response.status_code}")
                    else:
                        print(f"❌ Failed to send agent message: {agent_response.status_code}")
                else:
                    print(f"❌ Failed to take session: {take_response.status_code}")
            else:
                print(f"❌ Failed to get escalated sessions: {sessions_response.status_code}")
        else:
            print("❌ Session was not escalated")
    else:
        print(f"❌ Failed to create chat session: {chat_response.status_code}")
    
    print("\n" + "=" * 60)
    print("🏁 Escalation persistence test completed!")

if __name__ == "__main__":
    test_escalation_persistence() 