import fastapi
import uvicorn

from presentation.identify_person import PresentationIdentifyPerson


class ServerApp:
    def __init__(self):
        self.fastapi_app = fastapi.FastAPI()
        PresentationIdentifyPerson.setup(self.fastapi_app)

    def run(self):
        uvicorn.run(self.fastapi_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    app = ServerApp()
    app.run()
