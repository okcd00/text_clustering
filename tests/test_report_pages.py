import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_reports
class Links(HTMLParser):
    def __init__(self,text):super().__init__();self.ids=[];self.links=[];self.feed(text)
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for k in ['href','src']:
            if k in a:self.links.append(a[k])
class Reports(unittest.TestCase):
    def test_coverage_parity_provenance(self):
        items=build_reports.entries();self.assertEqual(len(items),9)
        self.assertEqual({x['source'] for x in items if x['source'].startswith('results/')},{p.relative_to(ROOT).as_posix() for p in (ROOT/'results').glob('*/README.md')})
        for name,s in build_reports.build_all().items():
            self.assertEqual(s,(ROOT/'docs/reports'/name).read_text())
        for x in items:
            s=(ROOT/'docs/reports'/f"{x['id']}.html").read_text()
            self.assertIn(hashlib.sha256((ROOT/x['source']).read_bytes()).hexdigest(),s)
            self.assertLess(s.index('report-conclusion'),s.index('report-body'))
            self.assertIn(build_reports.BASE+x['source'],s)
            self.assertIn(build_reports.BASE+x['reproduction'],s)
    def test_all_generated_local_links_and_fragments(self):
        for file in [ROOT/'docs/index.html',ROOT/'docs/survey.html',*(ROOT/'docs/reports').glob('*.html')]:
            parsed=Links(file.read_text());self.assertEqual(len(parsed.ids),len(set(parsed.ids)))
            for link in parsed.links:
                u=urlsplit(link)
                if link.startswith(build_reports.BASE):
                    self.assertTrue((ROOT/unquote(urlsplit(link[len(build_reports.BASE):]).path)).is_file(),link)
                if u.scheme or u.netloc:continue
                dest=(file.parent/unquote(u.path)).resolve() if u.path else file
                self.assertTrue(dest.is_file(),str((file,link)))
                if u.fragment and dest.suffix=='.html':self.assertIn(u.fragment,Links(dest.read_text()).ids,(file,link))
    def test_summary_not_duplicate_report(self):
        s=(ROOT/'docs/survey.html').read_text()
        for i in range(1,7):self.assertIn(f'id="roadmap-{i}"',s)
        for slug in ['calendar-paired-v1','calendar-paired-v2','calendar-gc-diagnostic-v1','hierarchy-e5-v1','hierarchy-sibling-preflight-v1']:self.assertIn('reports/'+slug+'.html',s)
        self.assertNotIn('69.31 / 68.50 MiB',s)
        self.assertNotIn('id="comparison-workbench" class="comparison-workbench" open',s)
        self.assertIn('id="comparison-chart"',s)
    def test_markdown_tables_code_and_links(self):
        body,_=build_reports.markdown(ROOT/'results/routing-v2/README.md')
        self.assertIn('<table>',body);self.assertIn('<pre><code>',body)
        self.assertNotIn('|---',body)
