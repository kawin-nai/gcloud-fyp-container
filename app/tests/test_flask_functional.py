from app.main import app
import unittest


class FlaskTest(unittest.TestCase):

    def test_verify_endpoint_valid(self):
        tester = app.test_client(self)
        response = tester.get("/verifyfromdb?camera=front")
        statuscode = response.status_code
        response_message = response.text
        self.assertEqual(response_message, 'Welcome to the face recognition API!')
        self.assertEqual(200, statuscode)



def endpoint_test():
    unittest.main()
