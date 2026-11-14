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

class ExtendedStreamingSection(unittest.TestCase):
    def test_v1_and_v2_stay_separate(self):
        content=(ROOT/'docs/survey.html').read_text()
        self.assertIn('id="roadmap-4"',content)
        self.assertIn('id="roadmap-5"',content)
        self.assertIn('calendar-paired-v1/results.json',content)
        self.assertIn('calendar-paired-v2/results.json',content)
        self.assertIn('5ms',content)
        self.assertIn('69.31 / 68.50 MiB',content)

class HierarchyPilotSection(unittest.TestCase):
    def test_cost_leads_and_preflight_remains_distinct(self):
        content=(ROOT/'docs/survey.html').read_text()
        self.assertIn('id="roadmap-6"',content)
        self.assertIn('hierarchy-e5-v1/results.json',content)
        self.assertIn('hierarchy-preflight-v1/README.md',content)
        self.assertIn('已知叶误拒率 57.4%',content)
        self.assertIn('132 / 230',content)
        self.assertIn('28 / 56；19 / 56',content)
        self.assertIn('38 / 48；32 / 48',content)
        self.assertIn('该历史预检本身只核验',content)
        section=content[content.index('id="roadmap-6"'):]
        self.assertLess(section.index('57.4%'),section.index('<table'))
