import cv2
import base64
import websockets
import json
import asyncio
import numpy as np
from datetime import datetime

class CameraApp:
    def __init__(self):
        self.camera_websocket_url = "ws://localhost:8000/camera_websocket"

    async def send_image(self):
        try:
            image = cv2.imread(
                "resources/data_set/osaka/input/gallery/1_0_5174.jpg")
            if image is None:
                print("画像の読み込みに失敗しました")
                return

            _, buffer = cv2.imencode('.jpg', image)
            b64_image = base64.b64encode(buffer).decode("utf-8")
            now = datetime.now()
            now_str = now.strftime("%Y%m%d_%H%M%S.%f")

            data = {
                "image": b64_image,
                "camera_id": 0,
                "view_id": 0,
                "timestamp": now_str
            }

            async with websockets.connect(self.camera_websocket_url) as websocket:
                await websocket.send(json.dumps(data))
                print("画像データを送信しました")

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続が閉じられました")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

    async def run(self):
        """メイン実行関数"""
        print("CameraAppを開始します...")
        await self.send_image()

    def run_sync(self):
        """同期版の実行関数"""
        asyncio.run(self.run())

if __name__ == "__main__":
    app = CameraApp()
    app.run_sync()
