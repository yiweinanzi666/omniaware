"""Preserve ambiguous header-like rows without inferring a new table schema."""
import json


def table_rows(text):
    lines = text.splitlines()
    assert len(lines) >= 2 and lines[1].startswith('Columns: ')
    headers = lines[1].removeprefix('Columns: ').split(' | ')
    rows = [line.split(' | ') for line in lines[2:]]
    assert all(len(headers) == len(cells) for cells in rows)
    if any(cells[0] == headers[0] for cells in rows):
        return '\n'.join([lines[0], lines[1], 'Complete source rows in original order:'] +
                         [f'Row {i}: ' + json.dumps(cells, ensure_ascii=False)
                          for i, cells in enumerate(rows, 1)])
    rendered = [lines[0], 'Each numbered line below is one table row.']
    for i, cells in enumerate(rows, 1):
        rendered.append(f'Row {i}: ' + '; '.join(h + '=' + json.dumps(v, ensure_ascii=False)
                                                for h, v in zip(headers, cells)))
    return '\n'.join(rendered)
