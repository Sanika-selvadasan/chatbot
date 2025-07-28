#!/usr/bin/env python3
"""
Test script for conversation summary feature
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_summary_feature():
    """Test the conversation summary feature"""
    
    print("🧪 Testing Conversation Summary Feature")
    print("=" * 50)
    
    # 1. Create a session with multiple messages
    print("\n1. Creating a session with multiple messages...")
    session_id = f"test_session_{int(time.time())}"
    
    # Send multiple messages to create a conversation
    messages = [
        "Hello, I'm looking for a day use hotel room",
        "I need it for tomorrow from 2pm to 6pm",
        "What's the price for a 4-hour stay?",
        "I want to book it now"
    ]
    
    for i, message in enumerate(messages):
        print(f"   Sending message {i+1}: {message}")
        chat_response = requests.post(f"{BASE_URL}/chat", json={
            "session_id": session_id,
            "message": message
        })
        
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            print(f"   ✅ Bot response: {chat_data['reply'][:50]}...")
            
            if chat_data['escalated']:
                print(f"   🚨 Session escalated to agent: {chat_data.get('agent_id')}")
                agent_id = chat_data['agent_id']
                break
        else:
            print(f"   ❌ Failed to send message: {chat_response.status_code}")
            return
    
    # 2. Test the summary endpoint
    print(f"\n2. Testing summary endpoint...")
    summary_response = requests.get(f"{BASE_URL}/session/{session_id}/summary")
    
    if summary_response.status_code == 200:
        summary_data = summary_response.json()
        print(f"✅ Summary endpoint working")
        print(f"✅ Message count: {summary_data['message_count']}")
        print(f"✅ Escalated: {summary_data['escalated']}")
        print(f"✅ Summary: {summary_data['summary']}")
    else:
        print(f"❌ Failed to get summary: {summary_response.status_code}")
        return
    
    # 3. Test the history endpoint
    print(f"\n3. Testing history endpoint...")
    history_response = requests.get(f"{BASE_URL}/session/{session_id}/history")
    
    if history_response.status_code == 200:
        history_data = history_response.json()
        print(f"✅ History endpoint working")
        print(f"✅ Total messages: {len(history_data['history'])}")
        
        # Show the conversation
        print(f"\n📝 Full Conversation:")
        for i, msg in enumerate(history_data['history']):
            role = msg['role'].capitalize()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"   {i+1}. {role}: {content}")
    else:
        print(f"❌ Failed to get history: {history_response.status_code}")
        return
    
    # 4. Test escalated sessions endpoint
    print(f"\n4. Testing escalated sessions endpoint...")
    sessions_response = requests.get(f"{BASE_URL}/agent/sessions")
    
    if sessions_response.status_code == 200:
        sessions_data = sessions_response.json()
        print(f"✅ Escalated sessions endpoint working")
        print(f"✅ Found {len(sessions_data['escalated_sessions'])} escalated sessions")
        
        # Find our session
        our_session = None
        for session in sessions_data['escalated_sessions']:
            if session['session_id'] == session_id:
                our_session = session
                break
        
        if our_session:
            print(f"✅ Our session found in escalated sessions")
            print(f"   - Agent ID: {our_session['agent_id']}")
            print(f"   - Message count: {our_session['message_count']}")
        else:
            print("❌ Our session not found in escalated sessions")
    else:
        print(f"❌ Failed to get escalated sessions: {sessions_response.status_code}")
    
    print("\n" + "=" * 50)
    print("🏁 Summary feature test completed!")

if __name__ == "__main__":
    test_summary_feature() 