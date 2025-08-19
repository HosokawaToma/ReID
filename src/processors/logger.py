import logging

class LoggerProcessor:
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('person_image_app.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
