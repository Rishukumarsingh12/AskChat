import logging

class CustomLogger:
    def __init__(self, log_file: str = "main.log"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                 '%(asctime)s - %(levelname)s - %(message)s',
                  datefmt='%Y-%m-%d %H:%M:%S'
            )

            file_handler.setFormatter(formatter)


            self.logger.addHandler(file_handler)

    def info(self,message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def error(self,message: str):
        self.logger.error(message)
        