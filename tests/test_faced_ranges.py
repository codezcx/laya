import unittest
from unittest.mock import MagicMock,patch
from eeg_tasks.faced_download import read_range
class RangeTransferTests(unittest.TestCase):
 def response(self,status=206,span='bytes 4-7/12',body=b'efgh'):
  r=MagicMock();r.status=status;r.headers={'Content-Range':span};r.read.return_value=body;r.__enter__.return_value=r;return r
 def test_exact_range(self):
  with patch('urllib.request.urlopen',return_value=self.response()) as fetch:
   self.assertEqual(read_range('https://example.org/object',4,7,12),b'efgh');self.assertEqual(fetch.call_args.args[0].headers['Range'],'bytes=4-7')
 def test_short_read_rejected_before_append(self):
  with patch('urllib.request.urlopen',return_value=self.response(body=b'ef')):
   with self.assertRaises(IOError):read_range('https://example.org/object',4,7,12)
 def test_ignored_or_wrong_range_rejected(self):
  for response in [self.response(status=200),self.response(span='bytes 0-3/12')]:
   with patch('urllib.request.urlopen',return_value=response):
    with self.assertRaises(ValueError):read_range('https://example.org/object',4,7,12)
if __name__=='__main__':unittest.main()
