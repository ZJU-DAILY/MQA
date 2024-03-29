class ResponseData:
    def __init__(self, code=200, message="", data=None):
        self.code = code
        self.message = message
        self.data = data

    def set_data(self, data):
        self.data = data

    def to_dict(self):
        response_dict = {
            "code": self.code,
            "message": self.message,
            "data": self.data
        }
        return response_dict

    def __str__(self):
        return f"ResponseData(code={self.code}, message='{self.message}', data={self.data})"
