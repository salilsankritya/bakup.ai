"""Test script for code-aware ingestion pipeline."""
import sys, os
sys.path.insert(0, '.')

from pathlib import Path
from core.ingestion.code_parser import (
    detect_language, parse_python, parse_javascript,
    parse_go, parse_json, parse_file,
)
from core.ingestion.code_chunker import chunk_file_code_aware, code_units_to_chunks
from core.ingestion.chunker import Chunk

# ==============================
# Test 1: Language detection
# ==============================
assert detect_language(Path('app.py'))       == 'python'
assert detect_language(Path('index.js'))     == 'javascript'
assert detect_language(Path('app.tsx'))      == 'tsx'
assert detect_language(Path('main.go'))      == 'go'
assert detect_language(Path('App.java'))     == 'java'
assert detect_language(Path('config.json'))  == 'json'
assert detect_language(Path('cfg.yaml'))     == 'yaml'
assert detect_language(Path('settings.toml'))== 'toml'
assert detect_language(Path('readme.md'))    == 'text'
print('[PASS] Language detection: all 9 cases correct')

# ==============================
# Test 2: Python parsing
# ==============================
py_code = (
    'import os\n'
    'from pathlib import Path\n'
    '\n'
    'class MyService:\n'
    '    """Service that does things."""\n'
    '    \n'
    '    def __init__(self, name):\n'
    '        self.name = name\n'
    '    \n'
    '    def process(self, data):\n'
    '        """Process the data."""\n'
    '        return data.upper()\n'
    '\n'
    'def helper_func(x, y):\n'
    '    """Add two numbers."""\n'
    '    return x + y\n'
    '\n'
    'MAX_RETRIES = 3\n'
)

units = parse_python(py_code)
kinds = [u.kind for u in units]
names = [u.name for u in units]

assert 'class' in kinds, f'No class found in {kinds}'
assert 'function' in kinds, f'No function found in {kinds}'
assert 'MyService' in names
assert 'helper_func' in names

# Check methods
methods = [u for u in units if u.kind == 'method']
assert len(methods) >= 2, f'Expected 2+ methods, got {len(methods)}'
method_names = [m.name for m in methods]
assert '__init__' in method_names
assert 'process' in method_names

# Check docstrings
service_class = [u for u in units if u.name == 'MyService'][0]
assert 'Service that does things' in service_class.docstring

helper = [u for u in units if u.name == 'helper_func'][0]
assert 'Add two numbers' in helper.docstring

# Check imports
assert any('import os' in imp for imp in helper.imports)

print(f'[PASS] Python parsing: {len(units)} units — {kinds}')

# ==============================
# Test 3: JavaScript parsing
# ==============================
js_code = (
    "import React from 'react';\n"
    "const axios = require('axios');\n"
    "\n"
    "class UserManager {\n"
    "    constructor() {\n"
    "        this.users = [];\n"
    "    }\n"
    "    \n"
    "    addUser(name) {\n"
    "        this.users.push(name);\n"
    "    }\n"
    "}\n"
    "\n"
    "function fetchData(url) {\n"
    "    return axios.get(url);\n"
    "}\n"
    "\n"
    "const processItem = async (item) => {\n"
    "    return item.toUpperCase();\n"
    "};\n"
    "\n"
    "export default UserManager;\n"
)

js_units = parse_javascript(js_code)
js_kinds = [u.kind for u in js_units]
js_names = [u.name for u in js_units]

assert 'class' in js_kinds, f'No class found: {js_kinds}'
assert 'function' in js_kinds, f'No function found: {js_kinds}'
assert 'UserManager' in js_names, f'UserManager not found: {js_names}'
assert 'fetchData' in js_names, f'fetchData not found: {js_names}'
assert 'processItem' in js_names, f'processItem not found: {js_names}'
print(f'[PASS] JavaScript parsing: {len(js_units)} units — {js_names}')

# ==============================
# Test 4: Go parsing
# ==============================
go_code = (
    'package main\n'
    '\n'
    'import (\n'
    '    "fmt"\n'
    '    "net/http"\n'
    ')\n'
    '\n'
    'type Server struct {\n'
    '    Port int\n'
    '    Host string\n'
    '}\n'
    '\n'
    'func (s *Server) Start() {\n'
    '    fmt.Println("Starting server")\n'
    '}\n'
    '\n'
    'func handleRequest(w http.ResponseWriter, r *http.Request) {\n'
    '    fmt.Fprintf(w, "Hello")\n'
    '}\n'
)

go_units = parse_go(go_code)
go_names = [u.name for u in go_units]
assert 'Server' in go_names, f'Server not found: {go_names}'
assert 'Start' in go_names, f'Start not found: {go_names}'
assert 'handleRequest' in go_names, f'handleRequest not found: {go_names}'
print(f'[PASS] Go parsing: {len(go_units)} units — {go_names}')

# ==============================
# Test 5: JSON parsing
# ==============================
json_code = '{\n  "name": "my-app",\n  "version": "1.0.0",\n  "dependencies": {\n    "express": "^4.18.0"\n  }\n}'

json_units = parse_json(json_code)
json_names = [u.name for u in json_units]
assert len(json_units) >= 1
print(f'[PASS] JSON parsing: {len(json_units)} units — {json_names}')

# ==============================
# Test 6: Real file parsing (session.py)
# ==============================
sample = Path('../sample-project/src/auth/session.py').read_text()
real_units = parse_file(sample, 'python')
real_names = [u.name for u in real_units]

assert 'SessionStore' in real_names, f'SessionStore not found in {real_names}'
func_units = [u for u in real_units if u.kind in ('function', 'method')]
assert len(func_units) >= 3, f'Expected 3+ functions/methods, got {len(func_units)}'
print(f'[PASS] Real file parsing: {len(real_units)} units — {real_names}')

# ==============================
# Test 7: Code chunker integration
# ==============================
chunks = chunk_file_code_aware(
    Path('../sample-project/src/auth/session.py'),
    Path('../sample-project'),
)
assert len(chunks) >= 1, f'No chunks produced'
# Verify chunks have metadata
for c in chunks:
    assert c.source_type == 'code'
    assert c.file_name == 'session.py'

# Check that at least one chunk has function_name or class_name
has_func = any(c.function_name for c in chunks)
has_class = any(c.class_name or (c.chunk_kind == 'class') for c in chunks)
assert has_func or has_class, 'No function/class metadata found in chunks'

print(f'[PASS] Code chunker: {len(chunks)} chunk(s) with metadata')
for c in chunks:
    kind = c.chunk_kind or 'unknown'
    name = c.function_name or c.class_name or '<module>'
    print(f'  {kind:12s} {name:20s} L{c.line_start}-L{c.line_end} ({c.language})')

# ==============================
# Test 8: Chunk metadata schema
# ==============================
c = chunks[0]
assert hasattr(c, 'language'), 'Missing language field'
assert hasattr(c, 'function_name'), 'Missing function_name field'
assert hasattr(c, 'class_name'), 'Missing class_name field'
assert hasattr(c, 'chunk_kind'), 'Missing chunk_kind field'
assert hasattr(c, 'docstring'), 'Missing docstring field'
assert hasattr(c, 'imports'), 'Missing imports field'
assert c.language == 'python'
print('[PASS] Chunk metadata schema: all fields present')

# ==============================
# Test 9: File walker with code-aware chunks
# ==============================
from core.ingestion.file_walker import walk_project
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

all_chunks = list(walk_project(Path('../sample-project')))
assert len(all_chunks) >= 1

code_chunks = [c for c in all_chunks if c.source_type == 'code']
log_chunks = [c for c in all_chunks if c.source_type == 'log']

print(f'[PASS] File walker: {len(all_chunks)} total chunk(s), '
      f'{len(code_chunks)} code, {len(log_chunks)} log')

# Check code chunks have language metadata
code_with_lang = [c for c in code_chunks if c.language]
print(f'  Code chunks with language metadata: {len(code_with_lang)}/{len(code_chunks)}')

# ==============================
# Test 10: Backward compatibility — old Chunk fields still work
# ==============================
from core.ingestion.chunker import chunk_file
old_chunks = chunk_file(
    Path('../sample-project/src/auth/session.py'),
    Path('../sample-project'),
    source_type='code',
)
assert len(old_chunks) >= 1
# Old chunks should still have default empty strings for new fields
for c in old_chunks:
    assert c.language == ''
    assert c.function_name == ''
print(f'[PASS] Backward compatibility: {len(old_chunks)} old-style chunk(s) still work')

print()
print('All 10 code ingestion tests passed.')
