from sse_starlette.sse import EventSourceResponse


class PresentationClient:
    @staticmethod
    async def endpoint():
        return EventSourceResponse()

