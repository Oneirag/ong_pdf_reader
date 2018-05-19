
import unittest
import os

from config_endesa import config_bill_of_lading
from process_bl_pdf_with_image_alignmet import parse_bl_with_alignment


class TestBillOfLading(unittest.TestCase):


    def test_pdfs(self):
        for file_name, parsed in config_bill_of_lading.items():
            pdf_name = os.path.join(os.path.dirname(__file__), file_name)
            parsed_text = parse_bl_with_alignment(pdf_name)
        self.assertEqual(parsed, parsed_text, f"Error in file {pdf_name}")

if __name__ == '__main__':
    unittest.main()