#!/usr/bin/env python3
"""Simple Flask server to power the frontend UI.

Endpoints:
  GET /           -> serves preprod-enterprise.html (static file)
  POST /api/upload -> accepts a .docx/.pdf file, runs the screenplay parser (rules or LLM if available), returns parsed scenes JSON
  GET /api/download?format={json,csv,xlsx} -> returns an export of the last parsed result

This is intentionally small and synchronous. For large files or production use, run parsing as a background job.
"""
from flask import Flask, request, jsonify, send_file, Response
from pathlib import Path
import tempfile
import os
import time
import io
import threading

ROOT = Path(__file__).resolve().parents[1]
APP = Flask(__name__, static_folder=str(ROOT), static_url_path='')

# We'll keep the last parsed data in memory for quick export/download.
_last_parsed = {
    'scenes': [],
    'df_rows': []
}

def _asdict(scene):
    # SceneMetadata -> dict
    if hasattr(scene, '__dict__'):
        d = scene.__dict__.copy()
        # Ensure lists are plain lists
        for k, v in d.items():
            if isinstance(v, (set, tuple)):
                d[k] = list(v)
        return d
    return scene


@APP.route('/')
def index():
    # Serve the HTML file from repo root
    html_path = ROOT / 'preprod-enterprise.html'
    if html_path.exists():
        return APP.send_static_file('preprod-enterprise.html')
    return Response('UI not found', status=404)


@APP.route('/api/upload', methods=['POST'])
def api_upload():
    """Accept uploaded file and return parsed scenes JSON."""
    from app.screenplay_parser import read_docx, ScenarioParser, create_production_table

    if 'file' not in request.files:
        return jsonify({'error': 'file required'}), 400

    f = request.files['file']
    filename = f.filename or f'upload_{int(time.time())}.docx'
    save_dir = ROOT / 'input'
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_path = save_dir / filename
    f.save(str(saved_path))

    # Use template selection and options
    template = request.form.get('template', 'basic')
    use_llm = request.form.get('use_llm', 'false').lower() in ('1', 'true', 'yes')

    # Determine file type and extract text accordingly
    suffix = saved_path.suffix.lower()
    text = ''
    try:
        if suffix == '.pdf':
            # Use pdfplumber to extract text
            try:
                import pdfplumber
            except Exception:
                return jsonify({'error': 'pdfplumber is not installed on the server; please install it'}), 500

            try:
                with pdfplumber.open(str(saved_path)) as pdf:
                    pages = [p.extract_text() or '' for p in pdf.pages]
                    text = '\n\n'.join(pages).strip()
            except Exception as e:
                return jsonify({'error': f'failed to read PDF: {e}'}), 500
        else:
            # default: try docx reader
            try:
                text = read_docx(str(saved_path))
            except Exception as e:
                return jsonify({'error': f'failed to read uploaded file: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'unexpected error reading file: {e}'}), 500

    parser = ScenarioParser(use_llm=use_llm)
    scenes = parser.parse_screenplay(text)

    rows = []
    for s in scenes:
        d = _asdict(s)
        rows.append(d)

    # Save in-memory for subsequent export
    _last_parsed['scenes'] = rows

    # Build tabular rows for quick client consumption
    df = create_production_table(scenes)
    rows_table = df.to_dict(orient='records')
    _last_parsed['df_rows'] = rows_table

    # Prepare simple stats
    stats = {
        'total_scenes': len(scenes),
        'unique_locations': len(df['Объект'].unique()) if 'Объект' in df.columns else 0,
        'top_characters': []
    }

    # Top characters
    all_chars = []
    for s in scenes:
        all_chars.extend(getattr(s, 'characters', []) or [])
    from collections import Counter
    stats['top_characters'] = [c for c, _ in Counter(all_chars).most_common(5)]

    return jsonify({'scenes': rows, 'table': rows_table, 'stats': stats})


@APP.route('/api/download')
def api_download():
    """Return last parsed data in requested format."""
    fmt = request.args.get('format', 'json').lower()
    from app.screenplay_parser import create_production_table

    if not _last_parsed['df_rows']:
        return jsonify({'error': 'no parsed data available; upload a file first'}), 400

    if fmt == 'json':
        return jsonify({'table': _last_parsed['df_rows']})

    if fmt == 'csv':
        import csv
        si = io.StringIO()
        # use keys from first row
        writer = csv.DictWriter(si, fieldnames=_last_parsed['df_rows'][0].keys())
        writer.writeheader()
        for r in _last_parsed['df_rows']:
            writer.writerow(r)
        output = si.getvalue().encode('utf-8')
        return Response(output, mimetype='text/csv', headers={
            'Content-Disposition': 'attachment; filename=production_table.csv'
        })

    if fmt == 'xlsx':
        # Reconstruct scenes -> DataFrame -> Excel bytes
        df = None
        try:
            # create DataFrame from rows
            import pandas as pd
            df = pd.DataFrame(_last_parsed['df_rows'])
        except Exception:
            df = None

        if df is None:
            return jsonify({'error': 'failed to build table for xlsx'}), 500

        tmp = io.BytesIO()
        try:
            # Use pandas ExcelWriter with openpyxl
            with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='КПП')
            tmp.seek(0)
            return send_file(tmp, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='production_table.xlsx')
        except Exception as e:
            return jsonify({'error': f'failed to generate xlsx: {e}'}), 500

    return jsonify({'error': 'unsupported format'}), 400


def run(port: int = 5000):
    APP.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    run()
