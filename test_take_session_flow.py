#!/usr/bin/env python3
"""
Test script for the complete take session flow
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_take_session_flow():
    """Test the complete take session flow"""
    
    print("🧪 Testing Complete Take Session Flow")
    print("=" * 50)
    
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
                
                # Find our session
                our_session = None
                for session in sessions_data['escalated_sessions']:
                    if session['agent_id'] == agent_id:
                        our_session = session
                        break
                
                if our_session:
                    print(f"✅ Our session found:")
                    print(f"   - Session ID: {our_session['session_id']}")
                    print(f"   - Agent ID: {our_session['agent_id']}")
                    print(f"   - Status: waiting")
                    
                    # 3. Take the session
                    print(f"\n3. Taking session...")
                    take_response = requests.post(f"{BASE_URL}/agent/sessions/{agent_id}/take")
                    
                    if take_response.status_code == 200:
                        take_data = take_response.json()
                        print(f"✅ Session taken successfully: {take_data['message']}")
                        
                        # 4. Check escalated sessions again (should be empty now)
                        print(f"\n4. Checking escalated sessions after taking...")
                        sessions_response_2 = requests.get(f"{BASE_URL}/agent/sessions")
                        
                        if sessions_response_2.status_code == 200:
                            sessions_data_2 = sessions_response_2.json()
                            print(f"✅ Remaining escalated sessions: {len(sessions_data_2['escalated_sessions'])}")
                            
                            # 5. Test agent chat
                            print(f"\n5. Testing agent chat...")
                            agent_message = "Hello! I'm a human agent here to help you with your booking request."
                            
                            agent_response = requests.post(f"{BASE_URL}/agent-chat", json={
                                "session_id": session_id,
                                "agent_id": agent_id,
                                "message": agent_message
                            })
                            
                            if agent_response.status_code == 200:
                                agent_data = agent_response.json()
                                print(f"✅ Agent message sent: {agent_data['message']}")
                                
                                # 6. Check session history
                                print(f"\n6. Checking session history...")
                                history_response = requests.get(f"{BASE_URL}/session/{session_id}/history")
                                
                                if history_response.status_code == 200:
                                    history_data = history_response.json()
                                    print(f"✅ Session history loaded: {len(history_data['history'])} messages")
                                    
                                    # Check for agent message
                                    agent_messages = [msg for msg in history_data['history'] if msg.get('role') == 'agent']
                                    if agent_messages:
                                        print(f"✅ Agent message found in history: {agent_messages[-1]['content']}")
                                    else:
                                        print("❌ Agent message not found in history")
                                else:
                                    print(f"❌ Failed to get session history: {history_response.status_code}")
                            else:
                                print(f"❌ Failed to send agent message: {agent_response.status_code}")
                        else:
                            print(f"❌ Failed to get escalated sessions after taking: {sessions_response_2.status_code}")
                    else:
                        print(f"❌ Failed to take session: {take_response.status_code}")
                else:
                    print("❌ Our session not found in escalated sessions")
            else:
                print(f"❌ Failed to get escalated sessions: {sessions_response.status_code}")
        else:
            print("❌ Session was not escalated")
    else:
        print(f"❌ Failed to create chat session: {chat_response.status_code}")
    
    print("\n" + "=" * 50)
    print("🏁 Take session flow test completed!")

if __name__ == "__main__":
    test_take_session_flow() 