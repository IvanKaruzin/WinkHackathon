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
import logging
import json

logger = logging.getLogger(__name__)

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
    """Accept uploaded file and store it for later processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'file required'}), 400

    f = request.files['file']
    filename = f.filename or f'upload_{int(time.time())}.docx'
    save_dir = ROOT / 'input'
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_path = save_dir / filename
    f.save(str(saved_path))

    # Store file info for processing
    _last_parsed['file_path'] = str(saved_path)
    _last_parsed['preset'] = request.form.get('template', 'full')
    _last_parsed['config_path'] = request.form.get('config', 'config.yaml')
    
    return jsonify({'status': 'file_uploaded', 'filename': filename})

@APP.route('/api/process', methods=['POST'])
def api_process():
    """Process the uploaded file and return parsed scenes JSON."""
    from screenplay_parser import read_docx, read_pdf, ScenarioParser, create_production_table
    
    if 'file_path' not in _last_parsed:
        return jsonify({'error': 'no file uploaded'}), 400

    saved_path = Path(_last_parsed['file_path'])
    preset = _last_parsed['preset']
    if preset not in ['basic', 'extended', 'full']:
        preset = 'full'
    
    config_path = _last_parsed['config_path']
    config_full_path = str(ROOT / config_path) if not os.path.isabs(config_path) else config_path

    # Determine file type and extract text
    suffix = saved_path.suffix.lower()
    text = ''
    try:
        if suffix == '.pdf':
            text = read_pdf(str(saved_path))
        else:
            text = read_docx(str(saved_path))
    except Exception as e:
        return jsonify({'error': f'failed to read file: {e}'}), 500

    # Initialize parser and process
    try:
        parser = ScenarioParser(config_path=config_full_path, preset=preset)
        scenes = parser.parse_screenplay(text)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Ошибка парсинга: {e}\n{error_trace}")
        return jsonify({'error': f'failed to parse screenplay: {e}'}), 500

    rows = []
    for s in scenes:
        d = _asdict(s)
        rows.append(d)

    # Save in-memory for subsequent export
    _last_parsed['scenes'] = rows

    # Build tabular rows for quick client consumption with preset
    df = create_production_table(scenes, preset=preset)
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

@APP.route('/api/progress')
def progress():
    """SSE endpoint for processing progress updates."""
    def generate():
        from screenplay_parser import SceneMetadata
        
        # Get processing stages from parser
        stages = [
            ("Загрузка файла", 10),
            ("Анализ структуры", 20),
            ("Извлечение сцен", 30),
            ("Обработка метаданных", 60),
            ("Формирование отчета", 90),
            ("Готово", 100)
        ]
        
        for stage, progress in stages:
            data = json.dumps({
                'progress': progress,
                'stage': stage,
                'message': f'{stage}...'
            })
            yield f"data: {data}\n\n"
            time.sleep(1)  # Simulate processing time
            
    return Response(generate(), mimetype='text/event-stream')

@APP.route('/api/download')
def api_download():
    """Return last parsed data in requested format."""
    fmt = request.args.get('format', 'json').lower()
    from screenplay_parser import create_production_table

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
