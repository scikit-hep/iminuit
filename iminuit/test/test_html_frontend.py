from unittest import TestCase
from iminuit.HtmlFrontend import HtmlFrontend

class TestHtmlFrontend(TestCase):
	
	def testConstructor(self):
		html_frontend = HtmlFrontend()

if __name__ == '__main__':
    unittest.main()
