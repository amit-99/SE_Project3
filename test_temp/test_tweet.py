import unittest
from unittest.mock import patch, Mock
from ..sentimental_analysis.realworld.views import tweetanalysis

class TweetAnalysisTestCase(unittest.TestCase):

    @patch('tweetanalysis.requests.get')
    def test_unsuccessful_http_request(self, mock_requests_get):
        mock_response = Mock()
        mock_response.status_code = 404

        mock_requests_get.return_value = mock_response

        request = Mock()
        request.method = 'POST'
        request.POST.get.return_value = "https://github.com/karthikmp5/C.E.L.T_pro"
        response = tweetanalysis(request)

        self.assertIn("Webpage not accessible or not found", response)

    def test_get_request(self):
        request = Mock()
        request.method = 'GET'
        response = tweetanalysis(request)
        self.assertIn("Enter the tweet url to be analysed!", response)

if __name__ == '__main__':
    unittest.main()
