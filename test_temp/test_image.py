import unittest
from ..sentimental_analysis.realworld.views import imageanalysis  # Import the function from your module
from django.core.files.uploadedfile import SimpleUploadedFile

class ImageAnalysisTestCase(unittest.TestCase):
    def test_imageanalysis(self):
        request = self.create_dummy_request()
        
        response = imageanalysis(request)
        
        self.assertEqual(response.status_code, 200) 
        self.assertIn(b"Sentiment:", response.content)  
        self.assertIn(b"Positive Score:", response.content)  
        self.assertIn(b"Negative Score:", response.content) 
        self.assertIn(b"Neutral Score:", response.content) 

    def create_dummy_request(self):
        from django.http import HttpRequest, QueryDict
        from django.conf import settings
        from django.core.files.uploadedfile import InMemoryUploadedFile

        request = HttpRequest()
        request.POST = QueryDict("")
        request.FILES = QueryDict("")

        image_content = b"Sample image content"
        image_file = InMemoryUploadedFile(
            None,
            "imgfile",
            "img.jpg",
            "image/jpeg",
            len(image_content),
            None,
        )
        image_file.write(image_content)
        image_file.seek(0)

        request.FILES['imgfile'] = image_file

        return request

if __name__ == '__main__':
    unittest.main()
