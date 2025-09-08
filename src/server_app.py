import fastapi
import uvicorn
import numpy as np
from processors.reid.clip import ClipReIDProcessor
from processors.post.assign_person_id import AssignPersonIdPostProcessor
import json
import base64
import cv2
from datetime import datetime


class ServerApp:
    def __init__(self, num_initialize_features: int = 100):
        self.fastapi_app = fastapi.FastAPI()
        self.clip_reid_processor = ClipReIDProcessor()
        self.device = self.clip_reid_processor.get_device()
        self.assign_person_id_processor = AssignPersonIdPostProcessor(
            device=self.device,
            num_random_features=num_initialize_features
        )
        self.setup_routes()
        self.active_camera_connections: list[fastapi.WebSocket] = []
        self.active_app_connections: list[fastapi.WebSocket] = []

    def setup_routes(self):
        self.fastapi_app.add_websocket_route(
            "/camera_websocket", self.camera_websocket_endpoint)
        self.fastapi_app.add_websocket_route(
            "/app_websocket", self.app_websocket_endpoint)

    async def camera_websocket_endpoint(self, websocket: fastapi.WebSocket):
        await websocket.accept()
        self.active_camera_connections.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                json_data = json.loads(data)
                await self.reserve_camera_data(json_data)
        except fastapi.WebSocketDisconnect:
            self.active_camera_connections.remove(websocket)

    async def reserve_camera_data(self, json_data: dict):
        print("受け取りました")
        image_base64 = json_data["image"]
        image = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = np.array(image)
        camera_id = json_data["camera_id"]
        view_id = json_data["view_id"]
        camera_timestamp = json_data["timestamp"]
        feature = self.clip_reid_processor.extract_feature(
            image, camera_id, view_id)
        person_id = self.assign_person_id_processor.assign_person_id(feature)
        id_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        send_data = {
            "camera_id": camera_id,
            "view_id": view_id,
            "person_id": person_id,
            "camera_timestamp": camera_timestamp,
            "id_timestamp": id_timestamp
        }
        print("送信します")
        for connection in self.active_app_connections:
            await connection.send_text(json.dumps(send_data))
        print("送信しました")

    async def app_websocket_endpoint(self, websocket: fastapi.WebSocket):
        await websocket.accept()
        self.active_app_connections.append(websocket)
        try:
            while True:
                await websocket.receive_text()
        except fastapi.WebSocketDisconnect:
            self.active_app_connections.remove(websocket)

    def run(self):
        uvicorn.run(self.fastapi_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    app = ServerApp(num_initialize_features=200000)
    app.run()
