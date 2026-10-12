(function () {
  'use strict';

  var CFG = window.DECLAS_CONFIG || {};
  var root = document.documentElement;
  root.classList.remove('no-js');

  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
    } catch (e) { return null; }
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  var LANG = /^fr/i.test(root.lang) ? 'fr' : 'en';
  // Pages in a sub-folder (fr/) set data-base="../" so shared assets resolve
  var BASE = root.getAttribute('data-base') || '';
  function asset(path) { return !path || /^(https?:|\/|data:)/.test(path) ? path : BASE + path; }
  function fmt(n) { return Number(n || 0).toLocaleString(LANG === 'fr' ? 'fr-FR' : 'en-US'); }

  var STR = {
    en: {
      tagline: 'Free, open-source desktop software that turns camera trap images and videos into analysis-ready data. Runs on your computer, no internet needed.',
      product: 'Product', download: 'Download', documentation: 'Documentation', installGuide: 'Installation guide',
      changelog: 'Changelog', releases: 'All releases', resources: 'Resources', video: '',
      modelExt: 'Model extensions', publish: 'Publish a model', faq: 'FAQ &amp; troubleshooting', cite: 'Cite Declas (Zenodo)',
      community: 'Community', bug: 'Report a bug', feature: 'Request a feature', contact: 'Contact', support: 'Support the project',
      related: 'Related', compare: 'Compare tools', privacy: 'Privacy', releasedUnder: 'Released under the', mit: 'MIT License',
      madeFor: 'Made for wildlife researchers and conservation teams',
      close: 'Close', dlTitle: 'Download Declas',
      dlIntro: 'Pick your operating system. Declas is free, open source, and works offline once installed.',
      soFar: 'downloads so far', winReq: '64-bit installer', macReq: 'Separate builds for Apple Silicon and Intel',
      linReq: '64-bit (x86-64) archive', dlWin: 'Download for Windows', dlLinux: 'Download for Linux', yourSystem: 'Your system',
      dlNote: 'Need help installing? Read the <a href="docs.html#install">installation guide</a>. Source code and older versions are on <a href="{gh}" target="_blank" rel="noopener">GitHub</a>.',
      formTitle: 'One quick step',
      formIntro: 'Tell us who you are so we can let you know when a new version of Declas is out. You will only be asked once on this computer.',
      back: 'Back to downloads', firstName: 'First name', lastName: 'Last name', email: 'Email address', country: 'Country',
      selectCountry: 'Select your country',
      privacyNote: 'Used only to email you about Declas updates. We never sell or share it, and you can ask us to delete it at any time. See our <a href="privacy.html">privacy notice</a>.',
      continueDl: 'Continue to download', starting: 'Starting your download…', started: 'Your download has started',
      notStarting: 'Not starting? <a href="#" data-direct-link>Use the direct link</a>.',
      installSteps: 'Installation steps', quickStart: 'Quick start guide', oneDl: '1 download', nDl: '{n} downloads',
      required: 'Required', badEmail: 'Enter a valid email address', moreText: 'Please write a little more',
      notAvailable: 'That file is not available yet.',
      sending: 'Sending…', sent: 'Thanks, your message was sent. We usually reply within a few days.',
      sendFail: 'We could not send it from here. <a href="{mailto}">Send it by email instead</a>.',
      holder: 'Account holder', bank: 'Bank', bankCode: 'Bank code', branchCode: 'Branch code', accountNumber: 'Account number',
      ribKey: 'RIB key', branch: 'Branch', reference: 'Payment reference', comingSoon: 'Details coming soon',
      qrSoon: 'QR code coming soon', qrAlt: 'Orabank PI QR code to donate to Declas, account GANDAHO STANISLAS MAHUSSI',
      piAlias: 'PI alias', copy: 'Copy', copied: 'Copied', copyWhat: 'Copy {x}', copyFail: 'Copy failed, please select the text.',
      depthClasses: 'distance in metres', detection: 'Detection', classification: 'Classification', depth: 'Depth',
      pagerInfo: '{a}–{b} of {n} models', prevPage: 'Previous page', nextPage: 'Next page',
      copyCode: 'Copy code', text: 'Text', shell: 'Shell', preview: 'Screenshot preview', anchor: 'Link to this section', mb: 'MB',
    },
    fr: {
      tagline: 'Logiciel de bureau libre et gratuit qui transforme les images et vidéos de pièges photographiques en données prêtes à analyser. Il fonctionne sur votre ordinateur, sans internet.',
      product: 'Produit', download: 'Télécharger', documentation: 'Documentation', installGuide: 'Guide d’installation',
      changelog: 'Journal des versions', releases: 'Toutes les versions', resources: 'Ressources', video: 'Tutoriel vidéo',
      modelExt: 'Extensions de modèles', publish: 'Publier un modèle', faq: 'FAQ et dépannage', cite: 'Citer Declas (Zenodo)',
      community: 'Communauté', bug: 'Signaler un bug', feature: 'Proposer une fonctionnalité', contact: 'Contact', support: 'Soutenir le projet',
      related: 'Liens utiles', compare: 'Comparer les outils', privacy: 'Confidentialité', releasedUnder: 'Publié sous', mit: 'licence MIT',
      madeFor: 'Conçu pour les chercheurs et les équipes de conservation de la faune',
      close: 'Fermer', dlTitle: 'Télécharger Declas',
      dlIntro: 'Choisissez votre système. Declas est gratuit, libre, et fonctionne hors ligne une fois installé.',
      soFar: 'téléchargements à ce jour', winReq: 'Installateur 64 bits', macReq: 'Versions distinctes pour Apple Silicon et Intel',
      linReq: 'Archive 64 bits (x86-64)', dlWin: 'Télécharger pour Windows', dlLinux: 'Télécharger pour Linux', yourSystem: 'Votre système',
      dlNote: 'Besoin d’aide pour l’installation ? Consultez le <a href="docs.html#install">guide d’installation</a>. Le code source et les anciennes versions sont sur <a href="{gh}" target="_blank" rel="noopener">GitHub</a>.',
      formTitle: 'Une petite étape',
      formIntro: 'Dites-nous qui vous êtes pour être prévenu de la sortie des nouvelles versions de Declas. Cette question ne vous sera posée qu’une fois sur cet ordinateur.',
      back: 'Retour aux téléchargements', firstName: 'Prénom', lastName: 'Nom', email: 'Adresse e-mail', country: 'Pays',
      selectCountry: 'Choisissez votre pays',
      privacyNote: 'Utilisées uniquement pour vous informer des mises à jour de Declas. Elles ne sont jamais vendues ni partagées, et vous pouvez demander leur suppression à tout moment. Voir notre <a href="privacy.html">politique de confidentialité</a>.',
      continueDl: 'Continuer vers le téléchargement', starting: 'Lancement du téléchargement…', started: 'Votre téléchargement a commencé',
      notStarting: 'Rien ne se passe ? <a href="#" data-direct-link>Utilisez le lien direct</a>.',
      installSteps: 'Étapes d’installation', quickStart: 'Guide de démarrage rapide', oneDl: '1 téléchargement', nDl: '{n} téléchargements',
      required: 'Obligatoire', badEmail: 'Saisissez une adresse e-mail valide', moreText: 'Merci d’en écrire un peu plus',
      notAvailable: 'Ce fichier n’est pas encore disponible.',
      sending: 'Envoi…', sent: 'Merci, votre message a bien été envoyé. Nous répondons généralement sous quelques jours.',
      sendFail: 'Impossible de l’envoyer d’ici. <a href="{mailto}">Envoyez-le plutôt par e-mail</a>.',
      holder: 'Titulaire du compte', bank: 'Banque', bankCode: 'Code banque', branchCode: 'Code guichet', accountNumber: 'Numéro de compte',
      ribKey: 'Clé RIB', branch: 'Agence', reference: 'Référence du paiement', comingSoon: 'Informations bientôt disponibles',
      qrSoon: 'QR code bientôt disponible', qrAlt: 'QR code PI Orabank pour faire un don à Declas, compte GANDAHO STANISLAS MAHUSSI',
      piAlias: 'Alias PI', copy: 'Copier', copied: 'Copié', copyWhat: 'Copier : {x}', copyFail: 'La copie a échoué, sélectionnez le texte.',
      depthClasses: 'distance en mètres', detection: 'Détection', classification: 'Classification', depth: 'Profondeur',
      pagerInfo: '{a}–{b} sur {n} modèles', prevPage: 'Page précédente', nextPage: 'Page suivante',
      copyCode: 'Copier le code', text: 'Texte', shell: 'Terminal', preview: 'Aperçu de la capture', anchor: 'Lien vers cette section', mb: 'Mo',
    },
  };
  function t(key, vars) {
    var s = STR[LANG][key] != null ? STR[LANG][key] : STR.en[key];
    if (vars) Object.keys(vars).forEach(function (k) { s = s.split('{' + k + '}').join(vars[k]); });
    return s;
  }

  var ICON = {
    windows: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 5.6 10.5 4.5v7H3zM11.5 4.35 21 3v8.5h-9.5zM3 12.5h7.5v7L3 18.4zM11.5 12.5H21V21l-9.5-1.35z"/></svg>',
    macos: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M16.37 12.6c-.02-2.2 1.8-3.26 1.88-3.31-1.03-1.5-2.62-1.7-3.18-1.73-1.35-.14-2.64.8-3.33.8-.69 0-1.74-.78-2.87-.76-1.47.02-2.83.86-3.59 2.18-1.53 2.66-.39 6.59 1.1 8.75.73 1.05 1.6 2.23 2.74 2.19 1.1-.04 1.51-.71 2.84-.71 1.32 0 1.7.71 2.86.69 1.18-.02 1.93-1.07 2.65-2.13.84-1.22 1.18-2.4 1.2-2.46-.03-.01-2.29-.88-2.3-3.5zM14.2 6.13c.6-.74 1.01-1.75.9-2.77-.87.04-1.93.58-2.55 1.31-.56.65-1.05 1.69-.92 2.68.97.08 1.96-.49 2.57-1.22z"/></svg>',
    linux: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.2c-2.1 0-3.4 1.7-3.4 4.1 0 1.2.2 2-.6 3.3-.9 1.4-2.6 3.6-2.6 6.2 0 .7.1 1.2.3 1.7-.7.3-1.5.7-1.5 1.5 0 1 1.3 1.1 2.4 1.6 1 .5 1.6 1.3 2.6 1.3.8 0 1.4-.5 1.7-1h2.2c.3.5.9 1 1.7 1 1 0 1.6-.8 2.6-1.3 1.1-.5 2.4-.6 2.4-1.6 0-.8-.8-1.2-1.5-1.5.2-.5.3-1 .3-1.7 0-2.6-1.7-4.8-2.6-6.2-.8-1.3-.6-2.1-.6-3.3 0-2.4-1.3-4.1-3.4-4.1zm-1.5 3.6c.4 0 .7.5.7 1.1 0 .4-.1.7-.3.9l-.6-.2c.1-.2.2-.4.2-.7 0-.3-.1-.5-.3-.5s-.3.3-.3.6v.2l-.5-.2v-.1c0-.6.4-1.1 1.1-1.1zm3 0c.6 0 1 .5 1 1.1 0 .2 0 .3-.1.5l-.5.1v-.2c0-.4-.2-.7-.4-.7s-.4.3-.4.6c0 .2 0 .3.1.4l-.5.2c-.1-.2-.2-.4-.2-.7 0-.7.4-1.3 1-1.3zm-1.5 2c.7 0 1.6.4 2.2.7.3.2.3.5 0 .7-.8.5-1.4.9-2.2.9s-1.4-.4-2.2-.9c-.3-.2-.3-.5 0-.7.6-.3 1.5-.7 2.2-.7z"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 20h16"/></svg>',
    close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>',
    back: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>',
    lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
  };

  /* Theme */

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    store('declas-theme', theme);
    var meta = $('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', theme === 'dark' ? '#15100C' : '#FBF7F0');
  }
  $all('.theme-toggle').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var dark = root.getAttribute('data-theme') === 'dark' ||
        (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
      setTheme(dark ? 'light' : 'dark');
    });
  });
  if (!root.getAttribute('data-theme')) {
    root.setAttribute('data-theme', matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }

  /* Language switch: remember the choice, stay on the same section */

  $all('.lang-switch [data-lang]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      store('declas-lang', a.getAttribute('data-lang'));
      if (a.getAttribute('aria-current') === 'true') { e.preventDefault(); return; }
      if (location.hash) a.setAttribute('href', a.getAttribute('href').split('#')[0] + location.hash);
    });
  });

  /* Mobile menu */

  var nav = $('.nav');
  var menuBtn = $('.menu-toggle');
  function closeMenu() {
    if (!nav) return;
    nav.classList.remove('open');
    if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
  }
  if (menuBtn && nav) {
    menuBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      var open = nav.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('click', function (e) {
      if (!nav.contains(e.target)) closeMenu();
    });
    nav.addEventListener('click', function (e) {
      if (e.target.closest('a, button')) closeMenu();
    });
  }

  /* Toast */

  var toastEl;
  var toastTimer;
  function toast(msg) {
    if (!toastEl) {
      toastEl = document.createElement('div');
      toastEl.className = 'toast';
      toastEl.setAttribute('role', 'status');
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove('show'); }, 2600);
  }

  /* Footer */

  var footerSlot = $('#site-footer');
  if (footerSlot) {
    var gh = 'https://github.com/' + CFG.repo;
    footerSlot.outerHTML =
      '<footer class="site-footer"><div class="container">' +
      '<div class="footer-grid">' +
      '<div class="footer-brand"><a class="brand" href="index.html"><img src="' + asset('assets/img/logo.png') + '" alt="" width="30" height="30">Declas</a>' +
      '<p>' + t('tagline') + '</p></div>' +
      '<div class="footer-col"><h4>' + t('product') + '</h4><ul>' +
      '<li><button type="button" data-open-download>' + t('download') + '</button></li>' +
      '<li><a href="docs.html">' + t('documentation') + '</a></li>' +
      '<li><a href="docs.html#install">' + t('installGuide') + '</a></li>' +
      '<li><a href="' + gh + '/blob/main/CHANGELOG.md" target="_blank" rel="noopener">' + t('changelog') + '</a></li>' +
      '<li><a href="' + gh + '/releases" target="_blank" rel="noopener">' + t('releases') + '</a></li>' +
      '</ul></div>' +
      '<div class="footer-col"><h4>' + t('resources') + '</h4><ul>' +
      (t('video') ? '<li><a href="' + esc(CFG.videoUrl) + '" target="_blank" rel="noopener">' + t('video') + '</a></li>' : '') +
      '<li><a href="docs.html#models">' + t('modelExt') + '</a></li>' +
      '<li><a href="docs.html#publish">' + t('publish') + '</a></li>' +
      '<li><a href="docs.html#faq">' + t('faq') + '</a></li>' +
      '<li><a href="' + esc(CFG.zenodoDoi) + '" target="_blank" rel="noopener">' + t('cite') + '</a></li>' +
      '</ul></div>' +
      '<div class="footer-col"><h4>' + t('community') + '</h4><ul>' +
      '<li><a href="' + gh + '" target="_blank" rel="noopener">GitHub</a></li>' +
      '<li><a href="' + gh + '/issues/new" target="_blank" rel="noopener">' + t('bug') + '</a></li>' +
      '<li><a href="' + gh + '/issues" target="_blank" rel="noopener">' + t('feature') + '</a></li>' +
      '<li><a href="contact.html">' + t('contact') + '</a></li>' +
      '<li><a href="donate.html">' + t('support') + '</a></li>' +
      '</ul></div>' +
      '<div class="footer-col"><h4>' + t('related') + '</h4><ul>' +
      '<li><a href="https://github.com/agentmorris/MegaDetector" target="_blank" rel="noopener">MegaDetector</a></li>' +
      '<li><a href="https://huggingface.co/depth-anything" target="_blank" rel="noopener">Depth Anything</a></li>' +
      '<li><a href="https://wildlabs.net" target="_blank" rel="noopener">WILDLABS</a></li>' +
      '<li><a href="about.html#compare">' + t('compare') + '</a></li>' +
      '<li><a href="privacy.html">' + t('privacy') + '</a></li>' +
      '</ul></div>' +
      '</div>' +
      '<div class="footer-bottom"><span>© ' + new Date().getFullYear() + ' Declas · ' + t('releasedUnder') + ' <a href="' + gh + '/blob/main/LICENSE" target="_blank" rel="noopener">' + t('mit') + '</a></span>' +
      '<span>' + t('madeFor') + '</span></div>' +
      '</div></footer>';
  }

  /* Backend */

  var hasBackend = !!(CFG.supabaseUrl && CFG.supabaseAnonKey);

  function rpc(fn, args) {
    // Accept the project URL with or without a trailing "/rest/v1/"
    var base = CFG.supabaseUrl.replace(/\/+$/, '').replace(/\/rest\/v1$/, '');
    return fetch(base + '/rest/v1/rpc/' + fn, {
      method: 'POST',
      headers: {
        apikey: CFG.supabaseAnonKey,
        Authorization: 'Bearer ' + CFG.supabaseAnonKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(args || {}),
    }).then(function (res) {
      return res.text().then(function (txt) {
        var data = txt ? JSON.parse(txt) : null;
        if (!res.ok) {
          var err = new Error((data && data.message) || ('Request failed (' + res.status + ')'));
          console.warn('Declas database call "' + fn + '" failed:', err.message);
          throw err;
        }
        return data;
      });
    });
  }

  function deviceId() {
    var id = store('declas-device');
    if (!id) {
      id = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() :
        'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
          var r = Math.random() * 16 | 0;
          return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
      store('declas-device', id);
    }
    return id;
  }

  var baseline = (CFG.downloadBaseline || []).reduce(function (s, b) { return s + (b.count || 0); }, 0);

  function demoStats() {
    var local = {};
    try { local = JSON.parse(store('declas-demo-downloads') || '{}'); } catch (e) { local = {}; }
    var by = { windows: local.windows || 0, macos: local.macos || 0, linux: local.linux || 0 };
    return { total: baseline + by.windows + by.macos + by.linux, windows: by.windows, macos: by.macos, linux: by.linux };
  }

  var api = {
    stats: function () {
      if (!hasBackend) return Promise.resolve(demoStats());
      return rpc('download_stats').catch(function () { return demoStats(); });
    },
    isKnown: function () {
      if (store('declas-registered') === '1') return Promise.resolve(true);
      if (!hasBackend) return Promise.resolve(false);
      return rpc('visitor_known', { p_device: deviceId() }).then(function (known) {
        if (known) store('declas-registered', '1');
        return !!known;
      }).catch(function () { return false; });
    },
    register: function (data) {
      store('declas-registered', '1');
      if (!hasBackend) {
        store('declas-demo-subscriber', JSON.stringify(data));
        return Promise.resolve();
      }
      var payload = {
        p_email: data.email, p_first_name: data.firstName, p_last_name: data.lastName,
        p_country: data.country, p_device: deviceId(), p_consent: true,
      };
      return rpc('register_subscriber', payload).catch(function (err) {
        // Keep the download flowing; send the sign-up again on the next visit.
        store('declas-pending-signup', JSON.stringify(payload));
        throw err;
      });
    },
    recordDownload: function (os, variant, version) {
      if (!hasBackend) {
        var local = {};
        try { local = JSON.parse(store('declas-demo-downloads') || '{}'); } catch (e) { local = {}; }
        local[os] = (local[os] || 0) + 1;
        store('declas-demo-downloads', JSON.stringify(local));
        return Promise.resolve(demoStats());
      }
      return rpc('record_download', { p_os: os, p_variant: variant, p_version: version, p_device: deviceId() })
        .catch(function () { return null; });
    },
    contact: function (data) {
      if (!hasBackend) return Promise.reject(new Error('offline'));
      return rpc('submit_contact', {
        p_name: data.name, p_email: data.email, p_topic: data.topic,
        p_message: data.message, p_website: data.website || '',
      });
    },
  };

  if (hasBackend) {
    var pendingSignup = store('declas-pending-signup');
    if (pendingSignup) {
      try {
        rpc('register_subscriber', JSON.parse(pendingSignup))
          .then(function () { store('declas-pending-signup', null); })
          .catch(function () {});
      } catch (e) { store('declas-pending-signup', null); }
    }
  }

  /* Counters */

  var lastStats = null;
  function paintStats(stats, bump) {
    if (!stats) return;
    lastStats = stats;
    $all('[data-count]').forEach(function (el) {
      var key = el.getAttribute('data-count');
      var value = stats[key];
      if (value == null) return;
      var text = fmt(value);
      if (el.textContent === text) return;
      el.textContent = text;
      if (bump) {
        el.classList.remove('bump');
        void el.offsetWidth;
        el.classList.add('bump');
      }
    });
    $all('[data-count-label]').forEach(function (el) {
      var n = stats[el.getAttribute('data-count-label')] || 0;
      el.textContent = n === 0 ? '' : n === 1 ? t('oneDl') : t('nDl', { n: fmt(n) });
    });
  }
  if ($('[data-count]') || $('[data-open-download]')) {
    api.stats().then(function (s) { paintStats(s, false); });
  }

  /* Release info */

  var release = null;
  function loadRelease() {
    if (release) return Promise.resolve(release);
    var cached = null;
    try { cached = JSON.parse(sessionStorage.getItem('declas-release') || 'null'); } catch (e) { cached = null; }
    if (cached) { release = cached; return Promise.resolve(release); }
    var fallback = fallbackRelease();
    return fetch('https://api.github.com/repos/' + CFG.repo + '/releases/latest')
      .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (data) {
        var out = { version: String(data.tag_name || CFG.version).replace(/^v/, ''), url: data.html_url, assets: {} };
        (data.assets || []).forEach(function (a) {
          var n = a.name;
          var key = /\.exe$/i.test(n) ? 'windows' :
            /arm64/i.test(n) && /\.dmg$/i.test(n) ? 'macos-arm' :
            /intel/i.test(n) && /\.dmg$/i.test(n) ? 'macos-intel' :
            /linux/i.test(n) && /\.tar\.gz$/i.test(n) ? 'linux' : null;
          if (key) out.assets[key] = { name: n, url: a.browser_download_url, size: a.size };
        });
        Object.keys(fallback.assets).forEach(function (k) { if (!out.assets[k]) out.assets[k] = fallback.assets[k]; });
        release = out;
        try { sessionStorage.setItem('declas-release', JSON.stringify(out)); } catch (e) {}
        return out;
      })
      .catch(function () { release = fallback; return fallback; });
  }
  function fallbackRelease() {
    var v = CFG.version;
    var out = { version: v, assets: {} };
    Object.keys(CFG.fallbackAssets || {}).forEach(function (k) {
      var name = CFG.fallbackAssets[k].replace('{v}', v);
      out.assets[k] = { name: name, url: 'https://github.com/' + CFG.repo + '/releases/download/v' + v + '/' + name, size: 0 };
    });
    return out;
  }
  if ($('[data-latest-tag], [data-latest-version], [data-latest-link]')) {
    loadRelease().then(function (rel) {
      $all('[data-latest-tag]').forEach(function (el) { el.textContent = 'v' + rel.version; });
      $all('[data-latest-version]').forEach(function (el) { el.textContent = rel.version; });
      if (rel.url) $all('[data-latest-link]').forEach(function (el) { el.href = rel.url; });
    });
  }
  function size(bytes) { return bytes ? Math.round(bytes / 1048576) + ' ' + t('mb') : ''; }

  function detectOS() {
    var ua = navigator.userAgent || '';
    var platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
    if (/Win/i.test(platform) || /Windows/i.test(ua)) return 'windows';
    if (/Mac/i.test(platform) || /Mac OS X/i.test(ua)) return /iPhone|iPad/i.test(ua) ? null : 'macos';
    if (/Linux/i.test(platform) && !/Android/i.test(ua)) return 'linux';
    return null;
  }

  /* Countries */

  var COUNTRY_CODES = 'AF AL DZ AD AO AG AR AM AU AT AZ BS BH BD BB BY BE BZ BJ BT BO BA BW BR BN BG BF BI CV KH CM CA CF TD CL CN CO KM CG CD CR CI HR CU CY CZ DK DJ DM DO EC EG SV GQ ER EE SZ ET FJ FI FR GA GM GE DE GH GR GD GT GN GW GY HT HN HK HU IS IN ID IR IQ IE IL IT JM JP JO KZ KE KI KW KG LA LV LB LS LR LY LI LT LU MG MW MY MV ML MT MH MR MU MX FM MD MC MN ME MA MZ MM NA NR NP NL NC NZ NI NE NG KP MK NO OM PK PW PS PA PG PY PE PH PL PT PR QA RE RO RU RW KN LC VC WS SM ST SA SN RS SC SL SG SK SI SB SO ZA KR SS ES LK SD SR SE CH SY TW TJ TZ TH TL TG TO TT TN TR TM TV UG UA AE GB US UY UZ VU VA VE VN YE ZM ZW'.split(' ');
  function countryOptions() {
    var names, english;
    try {
      names = new Intl.DisplayNames([LANG], { type: 'region' });
      english = new Intl.DisplayNames(['en'], { type: 'region' });
    } catch (e) { names = english = null; }
    // Show the name in the page language, but always store it in English
    var list = COUNTRY_CODES.map(function (c) {
      return { code: c, name: names ? names.of(c) : c, value: english ? english.of(c) : c };
    });
    list.sort(function (a, b) { return a.name.localeCompare(b.name, LANG); });
    return '<option value="" disabled selected>' + t('selectCountry') + '</option>' +
      list.map(function (c) { return '<option value="' + esc(c.value) + '">' + esc(c.name) + '</option>'; }).join('');
  }

  /* Download modal */

  var modal = null;
  var lastFocus = null;
  var pendingChoice = null;
  var knownVisitor = null;

  function buildModal() {
    modal = document.createElement('div');
    modal.className = 'modal';
    modal.id = 'download-modal';
    modal.setAttribute('aria-hidden', 'true');
    modal.innerHTML =
      '<div class="modal-backdrop" data-close></div>' +
      '<div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="dl-title">' +
      '<button class="icon-btn modal-close" type="button" data-close aria-label="' + t('close') + '">' + ICON.close + '</button>' +

      '<section class="view active" data-view="choose">' +
      '<div class="modal-head"><h2 id="dl-title">' + t('dlTitle') + ' <span class="muted" data-release-version style="font-weight:600"></span></h2>' +
      '<p>' + t('dlIntro') + '</p>' +
      '<div class="dl-total"><strong class="count" data-count="total">' + fmt(lastStats ? lastStats.total : baseline) + '</strong> ' + t('soFar') + '</div></div>' +
      '<div class="modal-body"><div class="os-grid">' +
      osCard('windows', 'Windows', t('winReq'), [{ key: 'windows', label: t('dlWin') }]) +
      osCard('macos', 'macOS', t('macReq'), [
        { key: 'macos-arm', label: 'Apple Silicon (M1+)' },
        { key: 'macos-intel', label: 'Intel', secondary: true },
      ]) +
      osCard('linux', 'Linux', t('linReq'), [{ key: 'linux', label: t('dlLinux') }]) +
      '</div><p class="dl-note">' + t('dlNote', { gh: 'https://github.com/' + esc(CFG.repo) + '/releases' }) + '</p></div>' +
      '</section>' +

      '<section class="view" data-view="form">' +
      '<div class="modal-head"><h2>' + t('formTitle') + '</h2><p>' + t('formIntro') + '</p></div>' +
      '<div class="modal-body">' +
      '<button class="back-link" type="button" data-back>' + ICON.back + t('back') + '</button>' +
      '<form class="form" id="signup-form" novalidate>' +
      '<div class="form-row">' +
      field('firstName', t('firstName'), 'text', 'given-name') +
      field('lastName', t('lastName'), 'text', 'family-name') +
      '</div>' +
      field('email', t('email'), 'email', 'email') +
      '<div class="field"><label for="f-country">' + t('country') + '</label><select class="select" id="f-country" name="country" required autocomplete="country-name">' + countryOptions() + '</select><span class="field-error" aria-live="polite"></span></div>' +
      '<div class="privacy-note">' + ICON.lock + '<span>' + t('privacyNote') + '</span></div>' +
      '<button class="btn btn-primary form-submit" type="submit">' + ICON.download + '<span data-submit-label>' + t('continueDl') + '</span></button>' +
      '<p class="form-status" role="status"></p>' +
      '</form></div>' +
      '</section>' +

      '<section class="view" data-view="done">' +
      '<div class="modal-body"><div class="done-state">' +
      '<div class="done-icon">' + ICON.check + '</div>' +
      '<h3>' + t('started') + '</h3>' +
      '<p data-done-file></p>' +
      '<p class="muted" style="font-size:.9rem">' + t('notStarting') + '</p>' +
      '<div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:22px">' +
      '<a class="btn btn-primary" data-install-link href="docs.html#install">' + t('installSteps') + '</a>' +
      '<a class="btn btn-secondary" href="docs.html#quick-start">' + t('quickStart') + '</a></div>' +
      '</div></div>' +
      '</section>' +
      '</div>';
    document.body.appendChild(modal);

    modal.addEventListener('click', function (e) {
      if (e.target.closest('[data-close]')) closeModal();
      if (e.target.closest('[data-back]')) showView('choose');
      var btn = e.target.closest('[data-asset]');
      if (btn) choose(btn.getAttribute('data-asset'));
    });
    modal.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeModal();
      if (e.key === 'Tab') trapFocus(e);
    });
    $('#signup-form', modal).addEventListener('submit', submitSignup);
    paintStats(lastStats, false);
  }

  function osCard(os, name, req, buttons) {
    var current = detectOS() === os;
    return '<article class="os-card' + (current ? ' is-current' : '') + '">' +
      (current ? '<span class="tag-current">' + t('yourSystem') + '</span>' : '') +
      '<div class="os-logo">' + ICON[os] + '</div>' +
      '<h3>' + name + '</h3>' +
      '<p class="os-file" data-file="' + buttons[0].key + '">&nbsp;</p>' +
      '<p class="os-count"><span data-count-label="' + os + '">' + (lastStats && lastStats[os] ? t('nDl', { n: fmt(lastStats[os]) }) : '') + '</span></p>' +
      '<div class="os-actions">' + buttons.map(function (b) {
        return '<button type="button" class="btn ' + (b.secondary ? 'btn-secondary' : 'btn-primary') + ' btn-block" data-asset="' + b.key + '">' +
          ICON.download + '<span>' + b.label + '</span></button>';
      }).join('') + '</div>' +
      '<p class="os-req">' + req + '</p>' +
      '</article>';
  }

  function field(name, label, type, autocomplete) {
    return '<div class="field"><label for="f-' + name + '">' + label + '</label>' +
      '<input class="input" id="f-' + name + '" name="' + name + '" type="' + type + '" required autocomplete="' + autocomplete + '"' +
      (type === 'email' ? ' inputmode="email" spellcheck="false"' : '') + ' maxlength="120">' +
      '<span class="field-error" aria-live="polite"></span></div>';
  }

  function paintRelease(rel) {
    if (!modal || !rel) return;
    var v = $('[data-release-version]', modal);
    if (v) v.textContent = 'v' + rel.version;
    $all('[data-file]', modal).forEach(function (el) {
      var a = rel.assets[el.getAttribute('data-file')];
      if (a) el.textContent = a.name + (a.size ? ' · ' + size(a.size) : '');
    });
  }

  function showView(name) {
    $all('.view', modal).forEach(function (v) { v.classList.toggle('active', v.getAttribute('data-view') === name); });
    var panel = $('.modal-panel', modal);
    panel.scrollTop = 0;
    var focusTarget = name === 'form' ? $('#f-firstName', modal) : $('.view.active button, .view.active a', modal);
    if (focusTarget) setTimeout(function () { focusTarget.focus({ preventScroll: true }); }, 30);
  }

  function openModal() {
    if (!modal) buildModal();
    lastFocus = document.activeElement;
    showView('choose');
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    loadRelease().then(paintRelease);
    api.stats().then(function (s) { paintStats(s, false); });
    if (knownVisitor === null) api.isKnown().then(function (k) { knownVisitor = k; });
    setTimeout(function () {
      var cur = $('.os-card.is-current [data-asset]', modal) || $('[data-asset]', modal);
      if (cur) cur.focus({ preventScroll: true });
    }, 60);
  }

  function closeModal() {
    if (!modal || !modal.classList.contains('open')) return;
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    if (lastFocus && lastFocus.focus) lastFocus.focus({ preventScroll: true });
  }

  function trapFocus(e) {
    var items = $all('.view.active button, .view.active a[href], .view.active input, .view.active select, .modal-close', modal)
      .filter(function (el) { return !el.disabled && el.offsetParent !== null; });
    if (!items.length) return;
    var first = items[0];
    var last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function choose(assetKey) {
    pendingChoice = assetKey;
    var go = function (known) {
      knownVisitor = known;
      if (known) startDownload(assetKey);
      else showView('form');
    };
    if (knownVisitor !== null) go(knownVisitor);
    else api.isKnown().then(go);
  }

  function validate(form) {
    var ok = true;
    $all('.field', form).forEach(function (f) {
      var input = $('input, select', f);
      var err = $('.field-error', f);
      var msg = '';
      var val = (input.value || '').trim();
      if (!val) msg = t('required');
      else if (input.type === 'email' && !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(val)) msg = t('badEmail');
      f.classList.toggle('invalid', !!msg);
      if (err) err.textContent = msg;
      if (msg && ok) { ok = false; input.focus(); }
    });
    return ok;
  }

  function submitSignup(e) {
    e.preventDefault();
    var form = e.target;
    if (!validate(form)) return;
    var btn = $('button[type="submit"]', form);
    var status = $('.form-status', form);
    btn.disabled = true;
    $('[data-submit-label]', form).textContent = t('starting');
    var data = {
      firstName: form.firstName.value.trim(),
      lastName: form.lastName.value.trim(),
      email: form.email.value.trim(),
      country: form.country.value,
    };
    api.register(data).catch(function () {}).then(function () {
      knownVisitor = true;
      btn.disabled = false;
      $('[data-submit-label]', form).textContent = t('continueDl');
      status.textContent = '';
      startDownload(pendingChoice);
    });
  }

  function startDownload(assetKey) {
    loadRelease().then(function (rel) {
      var asset = rel.assets[assetKey];
      if (!asset) { toast(t('notAvailable')); return; }
      var os = assetKey.indexOf('macos') === 0 ? 'macos' : assetKey;
      var variant = assetKey === 'macos-arm' ? 'arm64' : assetKey === 'macos-intel' ? 'intel' : 'x64';

      var a = document.createElement('a');
      a.href = asset.url;
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      a.remove();

      $('[data-done-file]', modal).textContent = asset.name + (asset.size ? ' (' + size(asset.size) + ')' : '');
      $('[data-direct-link]', modal).setAttribute('href', asset.url);
      $('[data-install-link]', modal).setAttribute('href', 'docs.html#install-' + os);
      showView('done');

      api.recordDownload(os, variant, rel.version).then(function (stats) {
        if (stats) paintStats(stats, true);
        else if (lastStats) {
          var next = Object.assign({}, lastStats);
          next.total += 1; next[os] = (next[os] || 0) + 1;
          paintStats(next, true);
        }
      });
    });
  }

  document.addEventListener('click', function (e) {
    var trigger = e.target.closest('[data-open-download]');
    if (trigger) { e.preventDefault(); openModal(); }
  });
  if (location.hash === '#download') openModal();

  /* Contact form */

  var contactForm = $('#contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', function (e) {
      e.preventDefault();
      var status = $('.form-status', contactForm);
      var ok = true;
      $all('.field', contactForm).forEach(function (f) {
        var input = $('input, select, textarea', f);
        if (!input || !input.required) return;
        var val = (input.value || '').trim();
        var msg = !val ? t('required') :
          input.type === 'email' && !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(val) ? t('badEmail') :
          input.name === 'message' && val.length < 10 ? t('moreText') : '';
        f.classList.toggle('invalid', !!msg);
        $('.field-error', f).textContent = msg;
        if (msg && ok) { ok = false; input.focus(); }
      });
      if (!ok) return;

      var data = {
        name: contactForm.name.value.trim(),
        email: contactForm.email.value.trim(),
        topic: contactForm.topic.value,
        message: contactForm.message.value.trim(),
        website: contactForm.website.value,
      };
      var btn = $('button[type="submit"]', contactForm);
      btn.disabled = true;
      status.className = 'form-status';
      status.textContent = t('sending');
      api.contact(data).then(function () {
        contactForm.reset();
        status.className = 'form-status ok';
        status.textContent = t('sent');
      }).catch(function () {
        var body = encodeURIComponent(data.message + '\n\n— ' + data.name + ' (' + data.email + ')');
        var subject = encodeURIComponent('[Declas] ' + data.topic);
        status.className = 'form-status err';
        status.innerHTML = t('sendFail', { mailto: 'mailto:' + esc(CFG.contactEmail) + '?subject=' + subject + '&amp;body=' + body });
      }).then(function () { btn.disabled = false; });
    });
  }

  /* Donation details */

  var bankList = $('#bank-details');
  if (bankList && CFG.donation) {
    var bank = CFG.donation.bank || {};
    var fr = LANG === 'fr';
    var rows = [
      [t('holder'), bank.accountName], [t('bank'), bank.bankName], ['IBAN', bank.iban], ['BIC / SWIFT', bank.swift],
      [t('bankCode'), bank.bankCode], [t('branchCode'), bank.branchCode], [t('accountNumber'), bank.accountNumber],
      [t('ribKey'), bank.ribKey], [t('branch'), (fr && bank.bankAddressFr) || bank.bankAddress],
      [t('reference'), (fr && bank.referenceFr) || bank.reference],
    ].filter(function (r) { return r[1] !== undefined; });
    bankList.innerHTML = rows.map(function (r) {
      var v = (r[1] || '').trim();
      // Group the IBAN by 4 for reading; copy it without spaces.
      var shown = r[0] === 'IBAN' ? v.replace(/(.{4})(?=.)/g, '$1 ') : v;
      return '<div class="kv-row"><dt>' + r[0] + '</dt>' +
        (v ? '<dd>' + esc(shown) + '</dd><button class="copy-btn" type="button" data-copy="' + esc(v) + '" aria-label="' + esc(t('copyWhat', { x: r[0] })) + '">' + t('copy') + '</button>'
          : '<dd class="pending">' + t('comingSoon') + '</dd><span></span>') + '</div>';
    }).join('');
    var rib = $('#rib-download');
    if (rib) {
      if (bank.ribPdf) rib.href = asset(bank.ribPdf);
      else rib.hidden = true;
    }

    var pio = CFG.donation.piorabank || {};
    var qr = $('#piorabank-qr');
    if (qr) {
      qr.innerHTML = pio.qrImage
        ? '<img src="' + esc(asset(pio.qrImage)) + '" alt="' + esc(t('qrAlt')) + '" width="240" height="290" data-zoom>'
        : '<span class="qr-empty">' + t('qrSoon') + '</span>';
    }
    var alias = $('#pi-alias');
    if (alias) {
      if (pio.alias) {
        alias.innerHTML = '<div class="kv-row"><dt>' + t('piAlias') + '</dt><dd>' + esc(pio.alias) + '</dd>' +
          '<button class="copy-btn" type="button" data-copy="' + esc(pio.alias) + '" aria-label="' + esc(t('copyWhat', { x: t('piAlias') })) + '">' + t('copy') + '</button></div>';
      } else {
        alias.hidden = true;
      }
    }
    var pioLink = $('#piorabank-link');
    if (pioLink) {
      if (pio.link) pioLink.href = pio.link;
      else pioLink.hidden = true;
    }
  }
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-copy]');
    if (!btn) return;
    var text = btn.getAttribute('data-copy');
    var done = function () {
      btn.textContent = t('copied');
      btn.classList.add('copied');
      setTimeout(function () { btn.textContent = t('copy'); btn.classList.remove('copied'); }, 1600);
    };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () { toast(t('copyFail')); });
  });

  /* Model registry (same file the app reads in Models > Extensions) */

  var REGISTRY_URL = 'https://raw.githubusercontent.com/' + CFG.repo + '/main/model_extensions/registry.json';
  var MODELS_PER_PAGE = 10;

  function loadRegistry() {
    var cached = null;
    try { cached = JSON.parse(sessionStorage.getItem('declas-registry') || 'null'); } catch (e) { cached = null; }
    if (cached) return Promise.resolve(cached);
    return fetch(REGISTRY_URL)
      .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (data) {
        var models = Array.isArray(data) ? data : (data.models || []);
        try { sessionStorage.setItem('declas-registry', JSON.stringify(models)); } catch (e) {}
        return models;
      });
  }

  // Registry regions are written in English; French pages show them translated when known
  var REGIONS_FR = {
    'Global': 'Mondial', 'Europe': 'Europe', 'Namibia': 'Namibie', 'New Zealand': 'Nouvelle-Zélande',
    'Colombia (Amazon)': 'Colombie (Amazonie)', 'Peru (Amazon)': 'Pérou (Amazonie)', 'Iran': 'Iran',
    'Southwest USA': 'Sud-ouest des États-Unis', 'Amazon': 'Amazonie', 'Serengeti (Tanzania)': 'Serengeti (Tanzanie)',
    'Sub-Saharan Africa': 'Afrique subsaharienne',
  };
  function region(name) { return (LANG === 'fr' && REGIONS_FR[name]) || name; }

  function link(text, url) {
    return url ? '<a href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(text) + '</a>' : esc(text);
  }

  function modelRow(m) {
    var task = String(m.task || '');
    var classes = m.classes || [];
    var classText = task.toLowerCase() === 'depth' ? t('depthClasses') :
      classes.length && classes.length <= 3 ? classes.join(', ') : String(classes.length || '–');
    return '<tr>' +
      '<td>' + link(m.display_name || m.name, m.info_url) + (m.author ? '<br><span class="muted" style="font-size:.8rem">' + esc(m.author) + '</span>' : '') + '</td>' +
      '<td>' + esc(t(task.toLowerCase()) || task) + '</td>' +
      '<td>' + esc(m.region ? region(m.region) : '–') + '</td>' +
      '<td>' + esc(classText) + '</td>' +
      '<td style="white-space:nowrap">' + (m.size_mb ? fmt(m.size_mb) + ' ' + t('mb') : '–') + '</td>' +
      '<td style="white-space:nowrap">' + link(m.license || '–', m.license_url) + '</td>' +
      '</tr>';
  }

  var modelTable = $('#model-table');
  var countEls = $all('[data-model-count]');
  if (modelTable || countEls.length) {
    loadRegistry().then(function (models) {
      if (!models.length) return;
      countEls.forEach(function (el) { el.textContent = fmt(models.length); });
      if (!modelTable) return;

      var body = $('tbody', modelTable);
      var pager = $('#model-pager');
      var pages = Math.ceil(models.length / MODELS_PER_PAGE);
      var page = 1;

      var render = function (scroll) {
        var start = (page - 1) * MODELS_PER_PAGE;
        var slice = models.slice(start, start + MODELS_PER_PAGE);
        body.innerHTML = slice.map(modelRow).join('');
        if (!pager) return;
        pager.hidden = pages < 2;
        var btns = '';
        for (var i = 1; i <= pages; i++) {
          btns += '<button type="button" class="pager-btn" data-page="' + i + '"' + (i === page ? ' aria-current="page"' : '') + '>' + i + '</button>';
        }
        pager.innerHTML =
          '<span class="pager-info">' + t('pagerInfo', { a: start + 1, b: start + slice.length, n: models.length }) + '</span>' +
          '<span class="pager-btns">' +
          '<button type="button" class="pager-btn" data-page="' + (page - 1) + '"' + (page === 1 ? ' disabled' : '') + ' aria-label="' + t('prevPage') + '">‹</button>' +
          btns +
          '<button type="button" class="pager-btn" data-page="' + (page + 1) + '"' + (page === pages ? ' disabled' : '') + ' aria-label="' + t('nextPage') + '">›</button>' +
          '</span>';
        if (scroll) {
          var top = modelTable.getBoundingClientRect().top;
          if (top < 80) window.scrollBy(0, top - 100);
        }
      };

      if (pager) {
        pager.addEventListener('click', function (e) {
          var btn = e.target.closest('[data-page]');
          if (!btn || btn.disabled) return;
          var next = Number(btn.getAttribute('data-page'));
          if (next < 1 || next > pages || next === page) return;
          page = next;
          render(true);
        });
      }
      render(false);
    }).catch(function () { /* keep the rows written in the page */ });
  }

  /* Code blocks: language label, copy button, syntax colours */

  var LANG_NAMES = { python: 'Python', json: 'JSON', bash: t('shell'), plaintext: t('text') };
  var codeBlocks = $all('.prose pre > code');
  codeBlocks.forEach(function (code) {
    var pre = code.parentNode;
    var lang = ((code.className.match(/language-(\w+)/) || [])[1]) || 'plaintext';
    var label = pre.getAttribute('data-file') || pre.getAttribute('data-lang') || LANG_NAMES[lang] || lang;
    var wrap = document.createElement('div');
    wrap.className = 'code-block';
    wrap.innerHTML = '<div class="code-head"><span class="code-lang">' + esc(label) + '</span>' +
      '<button class="copy-btn code-copy" type="button" aria-label="' + t('copyCode') + '">' + t('copy') + '</button></div>';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    $('.code-copy', wrap).addEventListener('click', function () {
      var btn = this;
      var text = code.textContent.replace(/\n$/, '');
      var done = function () {
        btn.textContent = t('copied');
        btn.classList.add('copied');
        setTimeout(function () { btn.textContent = t('copy'); btn.classList.remove('copied'); }, 1600);
      };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () { toast(t('copyFail')); });
    });
  });
  if (codeBlocks.length) {
    // highlight.js is loaded with "defer", so it is ready by DOMContentLoaded
    document.addEventListener('DOMContentLoaded', function () {
      if (!window.hljs) return;
      codeBlocks.forEach(function (code) {
        // Blocks holding live elements (e.g. the latest version number) keep their markup
        if (code.children.length || /language-plaintext/.test(code.className)) return;
        window.hljs.highlightElement(code);
      });
    });
  }

  /* Lightbox */

  var lightbox;
  document.addEventListener('click', function (e) {
    var img = e.target.closest('.shot img, [data-zoom]');
    if (!img) return;
    if (!lightbox) {
      lightbox = document.createElement('div');
      lightbox.className = 'lightbox';
      lightbox.setAttribute('role', 'dialog');
      lightbox.setAttribute('aria-label', t('preview'));
      lightbox.innerHTML = '<img alt="">';
      lightbox.addEventListener('click', function () { lightbox.classList.remove('open'); });
      document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape') lightbox.classList.remove('open'); });
      document.body.appendChild(lightbox);
    }
    var big = $('img', lightbox);
    big.src = img.currentSrc || img.src;
    big.alt = img.alt;
    lightbox.classList.add('open');
  });

  /* Reveal on scroll */

  var reveals = $all('.reveal');
  if ('IntersectionObserver' in window && reveals.length) {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { reveal(en.target); ro.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -60px 0px' });
    reveals.forEach(function (el) { ro.observe(el); });
  } else {
    reveals.forEach(reveal);
  }
  function reveal(el) {
    el.classList.add('in');
    // After the entrance (500ms + stagger), drop the stagger delay so hover responds at once
    setTimeout(function () { el.classList.add('settled'); }, 750);
  }
  /* Feature tags marquee */

  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    $all('.marquee').forEach(function (box) {
      var track = $('.marquee-track', box);
      if (!track) return;
      Array.prototype.slice.call(track.children).forEach(function (chip) {
        var copy = chip.cloneNode(true);
        copy.setAttribute('aria-hidden', 'true');
        track.appendChild(copy);
      });
      box.classList.add('is-running');
      // Constant speed (~40px per second) whatever the text length or language
      track.style.setProperty('--marquee-duration', (track.scrollWidth / 2 / 40) + 's');
    });
  }

  /* Docs: table of contents */

  var toc = $('.toc');
  if (toc) {
    var links = $all('.toc a');
    var byId = {};
    links.forEach(function (a) {
      var id = a.getAttribute('href').slice(1);
      (byId[id] = byId[id] || []).push(a);
    });
    $all('.toc-mobile a').forEach(function (a) {
      var id = a.getAttribute('href').slice(1);
      (byId[id] = byId[id] || []).push(a);
      a.addEventListener('click', function () { var d = a.closest('details'); if (d) d.open = false; });
    });
    var heads = $all('.prose h2[id], .prose h3[id]');
    heads.forEach(function (h) {
      var anchor = document.createElement('a');
      anchor.className = 'h-anchor';
      anchor.href = '#' + h.id;
      anchor.setAttribute('aria-label', t('anchor'));
      anchor.textContent = '#';
      h.appendChild(anchor);
    });
    var current = null;
    var setActive = function (id) {
      if (id === current) return;
      current = id;
      links.concat($all('.toc-mobile a')).forEach(function (a) { a.classList.remove('active'); });
      (byId[id] || []).forEach(function (a) { a.classList.add('active'); });
      var active = (byId[id] || [])[0];
      if (active && active.closest('.toc')) {
        var box = toc.getBoundingClientRect();
        var r = active.getBoundingClientRect();
        if (r.top < box.top + 40 || r.bottom > box.bottom - 40) toc.scrollTop += r.top - box.top - box.height / 3;
      }
      var label = $('.toc-mobile [data-current]');
      if (label && active) label.textContent = active.textContent;
    };
    var onScroll = function () {
      var y = window.scrollY + 120;
      var id = heads.length ? heads[0].id : null;
      for (var i = 0; i < heads.length; i++) {
        if (heads[i].offsetTop <= y) id = heads[i].id; else break;
      }
      if (id) setActive(id);
    };
    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () { onScroll(); ticking = false; });
    }, { passive: true });
    onScroll();
  }

  var topBtn = $('.back-to-top');
  if (topBtn) {
    window.addEventListener('scroll', function () {
      topBtn.classList.toggle('show', window.scrollY > 900);
    }, { passive: true });
    topBtn.addEventListener('click', function () { window.scrollTo({ top: 0 }); });
  }
})();
