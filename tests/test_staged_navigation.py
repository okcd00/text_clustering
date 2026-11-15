import unittest,re,subprocess
from pathlib import Path
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1]
class Structure(HTMLParser):
    def __init__(self,text):
        super().__init__();self.stack=[];self.panel_parents={};self.feed(text)
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if 'survey-panel' in a.get('class','').split():self.panel_parents[a['id']]=list(self.stack)
        if t=='details':self.stack.append(a.get('class',''))
    def handle_endtag(self,t):
        if t=='details':self.stack.pop()
class StagedNavigation(unittest.TestCase):
    def test_evidence_panels_outside_collapsed_map(self):
        s=(ROOT/'docs/survey.html').read_text();p=Structure(s)
        for name in ['profiles','compare','overview','methods','datasets','roadmap']:
            self.assertEqual(p.panel_parents[name],[])
        self.assertNotIn('class="page-links"',s)
        self.assertEqual(s.count('aria-label="Survey 章节"'),1)
        self.assertIn('路由专页与逐次明细',s)
        self.assertIn('已归档资料 · 不参与当前比较',s)
    def test_conclusions_before_map_and_icons_unchanged(self):
        s=(ROOT/'docs/index.html').read_text()
        self.assertLess(s.index('id="scenes"'),s.index('class="home-pathway"'))
        self.assertEqual(s.count('class="scene-card"'),4)
        self.assertEqual(s.count('class="scene-limit"'),4)
        base=subprocess.check_output(['git','show','70b7ef3:docs/home.css'],cwd=ROOT,text=True)
        current=(ROOT/'docs/home.css').read_text()
        pattern=r'data:image/webp;base64,[A-Za-z0-9+/=]+'
        self.assertEqual(re.findall(pattern,base),re.findall(pattern,current))
