import unittest
from pathlib import Path
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1]
class Page(HTMLParser):
    def __init__(self,text):
        super().__init__();self.links=[];self.ids=set();self.icons=[];self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='a':self.links.append(a)
        if tag=='span' and 'branch-icon' in a.get('class','').split():self.icons.append(a)
class Home(unittest.TestCase):
    def test_branches_and_targets(self):
        p=Page((ROOT/'docs/index.html').read_text());s=Page((ROOT/'docs/survey.html').read_text())
        branches=[a for a in p.links if 'path-branch' in a.get('class','')]
        self.assertEqual(len(branches),6)
        self.assertEqual(len(p.icons),6)
        for a in branches:
            page,anchor=a['href'].split('#');self.assertEqual(page.split('?')[0],'survey.html');self.assertIn(anchor,s.ids)
        icons_css=(ROOT/'docs/home.css').read_text()
        self.assertEqual(icons_css.count('data:image/webp;base64,'),1)
        self.assertEqual(icons_css.count('background-position:'),6)
        for i in p.icons:self.assertEqual(i['aria-hidden'],'true')
        for a in p.links:
            if a['href'].startswith('#'):self.assertIn(a['href'][1:],p.ids)
            elif not a['href'].startswith('https:'):self.assertTrue((ROOT/'docs'/a['href'].split('#')[0].split('?')[0]).is_file())
    def test_method_links_and_icon_cascade(self):
        source=(ROOT/'docs/index.html').read_text()
        home=Page(source)
        survey=Page((ROOT/'docs/survey.html').read_text())
        methods=[a['href'] for a in home.links if a.get('class')=='method-ref']
        self.assertEqual(len(methods),8)
        for link in methods:
            page,anchor=link.split('#')
            self.assertEqual(page,'survey.html')
            self.assertTrue(anchor.startswith('profiles-'))
            self.assertIn(anchor,survey.ids)
        self.assertGreaterEqual(source.count('class="scene-metric"'),9)
        import re
        css=(ROOT/'docs/home.css').read_text()
        for rule in re.findall(r'[^{}]+\{[^{}]+\}',css):
            if '.branch-icon' in rule.split('{')[0]:
                self.assertNotRegex(rule, r'\bbackground\s*:')
    def test_legacy_links_and_static_home(self):
        text=(ROOT/'docs/index.html').read_text()
        self.assertIn("if(location.hash && !['scenes','pathway-title'].includes(location.hash.slice(1)))",text)
        self.assertIn("['selection','leaderboard','status']",text)
        self.assertIn('同一条主干，不同的分叉。',text)
        self.assertIn('prefers-reduced-motion',(ROOT/'docs/home.css').read_text())
