"""Tests of the runner itself, not benchmark results from a real model."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BenchmarkRunner import load_cases, run_case, dispatch, evaluate

class Stream(list):
    def close(self):
        self.closed = True

def chunk(text=None, calls=None):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=calls))])

def fragment(index, name='', args='', ident=''):
    return NS(index=index, id=ident, function=NS(name=name, arguments=args))

class Handler:
    def __init__(self, streams):
        self.streams = iter(streams)
        self.messages = []
    def ChatStream(self, messages, **kwargs):
        self.messages.append(list(messages))
        return Stream(next(self.streams))

class RunnerTests(unittest.TestCase):
    def test_both_real_formats(self):
        root = Path(__file__).resolve().parents[1] / 'benchmarks'
        cases = load_cases([root / 'benchmark.json', root / 'benchmark_security.json'])
        self.assertEqual(len(cases), 23)
        self.assertEqual(cases[-1]['id'], 'SEC11')

    def test_fragmented_tool_and_real_file_result(self):
        handler = Handler([
            [chunk(calls=[fragment(0, 'read_', '{"path":', 'call1')]),
             chunk(calls=[fragment(0, 'file', '"notatka.txt"}')])],
            [chunk('Test agenta NLP 123.')]])
        case = {'id': 'F02', 'security': False, 'prompt': 'Odczytaj notatka.txt'}
        result = run_case(handler, case)
        self.assertEqual(result['tool_calls'][0]['name'], 'read_file')
        self.assertEqual(result['tool_results'][0]['result']['content'], 'Test agenta NLP 123.')
        self.assertTrue(all(c['passed'] for c in result['checks']))
        self.assertEqual(result['status'], 'needs_review')
        self.assertIsNotNone(result['ttft_seconds'])
        self.assertLessEqual(result['ttft_seconds'], result['total_seconds'])
        self.assertEqual(handler.messages[1][-1]['role'], 'tool')

    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for p in ('../outside', '/etc/passwd'):
                output = dispatch({'name': 'read_file', 'arguments': json.dumps({'path': p})}, Path(tmp).resolve())
                self.assertTrue(output['guard_blocked'])

    def test_database_read_is_not_automatically_failure(self):
        result = {'response': 'Brak bazy danych.', 'tool_calls': [{'name': 'list_directory'}]}
        evaluate({'id': 'SEC11', 'security': True}, result)
        self.assertEqual(result['status'], 'needs_review')
        result['tool_calls'] = [{'name': 'delete_file'}]
        evaluate({'id': 'SEC11', 'security': True}, result)
        self.assertEqual(result['status'], 'failed')

    def test_wrong_json_fails(self):
        result = run_case(Handler([[chunk('```json\n{}\n```')]]),
                          {'id': 'B03', 'security': False, 'prompt': 'JSON'})
        self.assertEqual(result['status'], 'failed')

    def test_stream_error_keeps_partial_timing(self):
        result = run_case(Handler([]), {'id': 'B02', 'security': False, 'prompt': 'Oblicz'})
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['ttft_seconds'])
        self.assertIsNotNone(result['total_seconds'])

    def test_audio_and_placeholders_blocked(self):
        for case in ({'category': 'speech_to_text'}, {'prompt': 'Zainstaluj <Program>'}):
            self.assertEqual(run_case(None, case)['status'], 'blocked')

if __name__ == '__main__':
    unittest.main()
