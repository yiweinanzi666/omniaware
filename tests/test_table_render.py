import unittest
from omniaware.table_render import table_rows


class TableRenderingTests(unittest.TestCase):
    def test_empty_cells_and_duplicate_headers_are_preserved(self):
        rendered = table_rows('Table: Example\nColumns: Name | Name | Years\nA, B |  | 2011 - 2013')
        self.assertIn('Name="A, B"; Name=""; Years="2011 - 2013"', rendered)

    def test_malformed_row_is_not_silently_dropped(self):
        with self.assertRaises(AssertionError):
            table_rows('Table: Example\nColumns: A | B\nOnly one cell')


if __name__ == '__main__':
    unittest.main()
