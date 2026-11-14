import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_news_calendar import parse,split,audit,stamp,external,ROOT
P={'id':'test','warmup_start':'2014-03-17','test_start':'2014-03-24','test_end_exclusive':'2014-03-31','warmup_max_rows':2,'per_scored_day_max_rows':2}
def row(i,date,text='Same title'):
    return {'id':i,'time':stamp(date),'text':text,'url':str(i),'publisher':'','category':'b','story':str(i%2)}
def test_half_open_boundaries_and_empty_calendar_windows():
    rows=[row(1,'2014-03-17'),row(2,'2014-03-24'),row(3,'2014-03-25'),row(4,'2014-03-31')]
    windows=split(rows,P)
    assert len(windows)==8
    assert [[r['id'] for r in w[3]] for w in windows[:3]]==[[1],[2],[3]]
    assert all(not w[3] for w in windows[3:])
def test_label_blind_selection_and_order():
    rows=[row(i,'2014-03-24',str(i)) for i in range(12)]
    a=split(rows,P)
    b=split([dict(r,story='changed',category='e') for r in reversed(rows)],P)
    assert [[r['id'] for r in w[3]] for w in a]==[[r['id'] for r in w[3]] for w in b]
    assert a[1][3]==sorted(a[1][3],key=lambda r:(r['time'],r['id']))
def test_first_title_view_does_not_change_learning():
    rows=[row(1,'2014-03-17','Ａ'),row(2,'2014-03-24','a'),row(3,'2014-03-25','New')]
    d=audit(rows,P)
    assert d['windows'][1]['selected_rows']==1
    assert d['windows'][1]['first_title_scored_rows']==0
    assert d['selected_scored_rows']==2 and d['first_title_scored_rows']==1
def test_strict_parser_and_private_output_boundaries():
    with pytest.raises(ValueError):parse(b'bad\trow\n')
    with pytest.raises(UnicodeDecodeError):parse(b'\xff')
    with pytest.raises(ValueError):external(ROOT/'outputs')
