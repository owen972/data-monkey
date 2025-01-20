import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import io
from data_monkey import process_text_with_model, process_pdf, main

class TestDataMonkey(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock()
        self.mock_response = Mock()
        self.mock_response.choices = [Mock(message=Mock(content="Test summary"))]
        self.mock_client.chat.completions.create.return_value = self.mock_response

    def test_process_text_with_model_success(self):
        test_text = "Sample text for processing"
        result = process_text_with_model(test_text, self.mock_client)
        
        self.assertEqual(result, "Test summary")
        self.mock_client.chat.completions.create.assert_called_once()

    @patch('PyPDF2.PdfReader')
    def test_process_pdf_success(self, mock_pdf_reader):
        # Setup mock PDF content
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        result = process_pdf("test.pdf", self.mock_client)
        
        self.assertEqual(result, "Test summary")
        mock_page.extract_text.assert_called_once()

    def test_process_pdf_file_not_found(self):
        result = process_pdf("nonexistent.pdf", self.mock_client)
        self.assertIsNone(result)

    def test_process_pdf_general_exception(self):
        with patch('PyPDF2.PdfReader', side_effect=Exception("Test error")):
            result = process_pdf("test.pdf", self.mock_client)
            self.assertIsNone(result)

    @patch('openai.OpenAI')
    def test_main_success(self, mock_openai):
        mock_openai.return_value = self.mock_client
        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called_with("Test summary")

    @patch('openai.OpenAI')
    def test_main_failure(self, mock_openai):
        mock_openai.return_value = self.mock_client
        with patch('process_pdf', side_effect=Exception("Test error")):
            with patch('builtins.print') as mock_print:
                main()
                mock_print.assert_called_with("Error accessing the PDF file: Test error")

if __name__ == '__main__':
    unittest.main()
