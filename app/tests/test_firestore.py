from mockfirestore import MockFirestore
import unittest


class TestFirestore(unittest.TestCase):
    def test_client(self):
        db = MockFirestore()
        self.assertIsNotNone(db)

    def test_collection(self):
        db = MockFirestore()
        collection = db.collection('users')
        self.assertIsNotNone(collection)

    def test_document(self):
        db = MockFirestore()
        collection = db.collection('users')
        document = collection.document('user1')
        self.assertIsNotNone(document)

    def test_set(self):
        db = MockFirestore()
        collection = db.collection('users')
        document = collection.document('user1')
        document.set({'name': 'John'})
        self.assertEqual(document.get().to_dict(), {'name': 'John'})

    def test_get(self):
        db = MockFirestore()
        collection = db.collection('users')
        document = collection.document('user1')
        document.set({'name': 'John'})
        self.assertEqual(document.get().to_dict(), {'name': 'John'})

    def test_update(self):
        db = MockFirestore()
        collection = db.collection('users')
        document = collection.document('user1')
        document.set({'name': 'John'})
        document.update({'name': 'Jane'})
        self.assertEqual(document.get().to_dict(), {'name': 'Jane'})


def firestore_test():
    unittest.main()
