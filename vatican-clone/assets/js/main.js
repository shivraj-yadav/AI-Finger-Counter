(function () {
  const toggle = document.querySelector('.nav-toggle');
  const list = document.getElementById('primary-nav');
  if (!toggle || !list) return;
  toggle.addEventListener('click', () => {
    const expanded = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', String(!expanded));
    list.setAttribute('aria-expanded', String(!expanded));
  });
})();