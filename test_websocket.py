#!/usr/bin/env python3
"""
Test script for WebSocket functionality
"""

import asyncio
import websockets
import json
import time

async def test_websocket_chat():
    """Test WebSocket chat functionality"""
    
    print("🧪 Testing WebSocket Chat Functionality")
    print("=" * 50)
    
    session_id = f"test_session_{int(time.time())}"
    agent_id = f"agent_{int(time.time())}"
    
    # Test customer WebSocket connection
    print(f"\n1. Testing customer WebSocket connection for session {session_id}...")
    
    try:
        async with websockets.connect(f"ws://localhost:8000/ws/session/{session_id}") as websocket:
            print("✅ Customer WebSocket connected")
            
            # Send a test message
            test_message = {
                "type": "user_message",
                "message": "I want to book a room for tomorrow"
            }
            
            await websocket.send(json.dumps(test_message))
            print("✅ Test message sent")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            print(f"✅ Response received: {response_data['type']}")
            
            if response_data.get('escalated'):
                print(f"✅ Session escalated to agent: {response_data.get('agent_id')}")
                
                # Test agent WebSocket connection
                print(f"\n2. Testing agent WebSocket connection for agent {agent_id}...")
                
                try:
                    async with websockets.connect(f"ws://localhost:8000/ws/agent/{agent_id}") as agent_websocket:
                        print("✅ Agent WebSocket connected")
                        
                        # Send agent message
                        agent_message = {
                            "type": "agent_message",
                            "session_id": session_id,
                            "message": "Hello! I'm a human agent here to help you with your booking."
                        }
                        
                        await agent_websocket.send(json.dumps(agent_message))
                        print("✅ Agent message sent")
                        
                        # Wait for confirmation
                        agent_response = await asyncio.wait_for(agent_websocket.recv(), timeout=10.0)
                        agent_response_data = json.loads(agent_response)
                        print(f"✅ Agent response: {agent_response_data['type']}")
                        
                except Exception as e:
                    print(f"❌ Agent WebSocket error: {e}")
            else:
                print("ℹ️ Session not escalated (this is normal for some messages)")
                
    except Exception as e:
        print(f"❌ Customer WebSocket error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 WebSocket test completed!")

if __name__ == "__main__":
    asyncio.run(test_websocket_chat()) 