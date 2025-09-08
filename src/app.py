import websockets
import asyncio
import json
from datetime import datetime

class App:
    def __init__(self):
        self.app_websocket_url = "ws://localhost:8000/app_websocket"

    async def receive_response(self):
        async with websockets.connect(self.app_websocket_url) as websocket:
            response = await websocket.recv()
            data = json.loads(response)
            print(data)
            app_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
            print(app_timestamp)

    async def run(self):
        await self.receive_response()

    def run_sync(self):
        print("Appを開始します...")
        asyncio.run(self.run())

if __name__ == "__main__":
    app = App()
    app.run_sync()
