import unittest, importlib.util
from pathlib import Path
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self): super().__init__();self.ids=[];self.urls=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for key in ('href','src'):
            if key in a:self.urls.append(a[key])
class Survey(unittest.TestCase):
    def test_generated_and_links(self):
        spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts/build_survey.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        content=(ROOT/'docs/survey.html').read_text();self.assertEqual(content,m.build())
        p=Links();p.feed(content);self.assertEqual(len(p.ids),len(set(p.ids)))
        for u in p.urls:
            if u.startswith('#'):self.assertIn(u[1:],p.ids)
            elif not u.startswith('http'):self.assertTrue((ROOT/'docs'/u.split('?')[0].split('#')[0]).exists(),u)

class StreamingSection(unittest.TestCase):
    def test_independent_new_table_and_closed_history(self):
        content=(ROOT/'docs/survey.html').read_text()
        self.assertIn('id="roadmap-4"',content)
        self.assertIn('calendar-paired-v1',content)
        self.assertIn('<th scope="col">维护方法</th>',content)
        self.assertIn('<details class="metric-help">\n<summary>查看历史场景 18',content)
        self.assertNotIn('<details class="metric-help" open>',content)
        self.assertNotIn('未运行编码或模型',content)
        self.assertIn('不加入静态与路由的原 42 个运行结果',content)
        # The previous charts retain their own dataset/runner, untouched by this table.
        self.assertIn('href="unified-results.json"',content)
