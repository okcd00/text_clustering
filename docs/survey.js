(() => {
  const panels = [...document.querySelectorAll('.survey-panel')];
  const tabs = [...document.querySelectorAll('[data-section]')];
  const input = document.querySelector('#survey-search');
  const status = document.querySelector('#search-status');
  function filter() {
    const q = input.value.trim().toLocaleLowerCase();
    const panel = panels.find(p => !p.hidden);
    const cards = [...panel.querySelectorAll('.evidence-card')];
    let count = 0;
    cards.forEach(card => { card.hidden = !!q && !card.textContent.toLocaleLowerCase().includes(q); if (!card.hidden) count++; });
    status.textContent = cards.length ? `本章节 ${count} / ${cards.length} 张卡片${q && !count ? ' · 没有匹配项，请换个关键词' : ''}；协议正文始终保留` : '本章节为阅读说明；方法与数据章节支持卡片筛选';
  }
  function route() {
    const id = location.hash.slice(1) || 'compare';
    const key = id.split('-')[0];
    const active = panels.find(p => p.id === key) || panels.find(p => p.id === 'compare');
    panels.forEach(p => p.hidden = p !== active);
    tabs.forEach(a => { if (a.dataset.section === active.id) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current'); });
    filter();
    if (id.includes('-')) requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
  }
  input.addEventListener('input', filter);
  document.querySelector('#clear-search').addEventListener('click', () => { input.value=''; filter(); input.focus(); });
  window.addEventListener('hashchange', route);
  route();
})();
