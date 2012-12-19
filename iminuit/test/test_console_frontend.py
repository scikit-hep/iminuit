from unittest import TestCase
from iminuit.ConsoleFrontend import ConsoleFrontend

class TestConsoleFrontend(TestCase):
	
	def testConstructor(self):
		console_frontend = ConsoleFrontend()

if __name__ == '__main__':
    unittest.main()
