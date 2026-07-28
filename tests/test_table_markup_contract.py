from __future__ import annotations

from app.models.schema import TableModel
from app.services.table_markup import parse_table_rows


def test_html_table_parser_preserves_valid_spans_and_defaults_invalid_spans() -> None:
    rows = parse_table_rows(
        '<table><tr><th rowspan="2" colspan="3">Heading</th>'
        '<th rowspan="0" colspan="bad">Other</th>'
        '<th rowspan="999999999999999999999" colspan="1001">Huge</th>'
        "</tr></table>"
    )

    assert rows[0][0].rowspan == 2
    assert rows[0][0].colspan == 3
    assert rows[0][1].rowspan == 1
    assert rows[0][1].colspan == 1
    assert rows[0][2].rowspan == 1
    assert rows[0][2].colspan == 1


def test_table_schema_remains_backward_compatible_without_geometry_fields() -> None:
    table = TableModel.model_validate(
        {
            "id": "legacy-table",
            "page_numbers": [1],
            "headers": ["Header"],
            "rows": [["Value"]],
            "cells": [[{"text": "Value", "rowspan": 1, "colspan": 1}]],
        }
    )

    assert table.header_cells == []
    assert table.cells[0][0].row_index is None
    assert table.cells[0][0].polygon == []
    assert table.cells[0][0].extraction_metadata == {}
