import edge_tts
import asyncio
import os

async def test_tts():
    try:
        print("Testing edge-tts connection...")
        communicate = edge_tts.Communicate("Hello, this is a test.", voice="en-GB-RyanMultilingualNeural")
        await communicate.save("test_output.mp3")
        
        if os.path.exists("test_output.mp3"):
            size = os.path.getsize("test_output.mp3")
            print(f"✓ Success! Audio file created: {size} bytes")
        else:
            print("✗ File was not created")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Check your internet connection - edge-tts needs to connect to Microsoft's servers")

asyncio.run(test_tts())
