from app.main import app
import unittest


class FlaskTest(unittest.TestCase):

    # Check for response 200
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get("/")
        statuscode = response.status_code
        response_message = response.text
        self.assertEqual(statuscode, 200)
        self.assertEqual(response_message, 'Welcome to the face recognition API!')

    # Check for invalid endpoint
    def test_invalid_endpoint(self):
        tester = app.test_client(self)
        response = tester.get("/random")
        statuscode = response.status_code
        self.assertEqual(statuscode, 404)

    # Check for input with invalid parameters
    def test_verify(self):
        tester = app.test_client(self)
        response = tester.get("/verifyfromdb")
        statuscode = response.status_code
        self.assertEqual(statuscode, 400)

    # Check for upload with invalid parameters
    def test_upload(self):
        tester = app.test_client(self)
        response = tester.post("/uploadtodb/random.jpg")
        statuscode = response.status_code
        self.assertEqual(statuscode, 400)

    # Check for upload with wrong endpoint
    def test_upload_with_wrong_endpoint(self):
        tester = app.test_client(self)
        response = tester.get("/uploadtodb/random.jpg")
        statuscode = response.status_code
        self.assertEqual(statuscode, 405)


def endpoint_test():
    unittest.main()
