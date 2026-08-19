const ZENODO_COUNT = 270;
const REPO = 'stangandaho/declas';

// Replace with your actual YouTube video URL
const VIDEO_URL = 'https://youtu.be/6sAMYketcKs';

const OS_LABELS = {
  'windows': 'Windows',
  'macos-arm': 'macOS (Apple Silicon)',
  'macos-intel': 'macOS (Intel)',
  'linux': 'Linux',
};

let downloadLinks = {};
let selectedOS = detectOS();

/* OS detection */

function detectOS() {
  const ua = navigator.userAgent;
  if (/Windows/i.test(ua)) return 'windows';
  if (/Mac/i.test(ua)) return 'macos-arm';   // default ARM; user can switch to Intel
  if (/Linux/i.test(ua)) return 'linux';
  return 'windows';
}

/* GitHub API */

async function loadRelease() {
  try {
    const res = await fetch(`https://api.github.com/repos/${REPO}/releases`);
    const releases = await res.json();
    if (!Array.isArray(releases) || releases.length === 0) return;

    let ghCount = 0;

    releases.forEach((release, idx) => {
      (release.assets || []).forEach(asset => {
        ghCount += asset.download_count;

        if (idx === 0) {
          const name = asset.name;
          if (/\.exe$/i.test(name)) downloadLinks['windows'] = asset.browser_download_url;
          else if (/arm64/i.test(name)) downloadLinks['macos-arm']   = asset.browser_download_url;
          else if (/intel/i.test(name)) downloadLinks['macos-intel'] = asset.browser_download_url;
          else if (/linux.*\.tar\.gz$/i.test(name)) downloadLinks['linux'] = asset.browser_download_url;
        }
      });
    });

    const latest = releases[0];
    if (latest && latest.tag_name) {
      document.getElementById('release-tag').textContent = `Version ${latest.tag_name}`;
    }

    const total = ZENODO_COUNT + ghCount;
    document.getElementById('counter-num').textContent = total.toLocaleString();
  } catch (_) {
    document.getElementById('counter-num').textContent = ZENODO_COUNT.toLocaleString();
  }
}

/* Download button */

document.getElementById('download-btn').addEventListener('click', () => {
  const url = downloadLinks[selectedOS];
  const a = document.createElement('a');
  a.href = url || `https://github.com/${REPO}/releases/latest`;
  if (url) a.download = '';
  else a.target = '_blank';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

/* Video Docs button */

document.getElementById('video-btn').addEventListener('click', () => window.open(VIDEO_URL, '_blank', 'noopener,noreferrer'));

/* Dropdown */

const dropdown = document.getElementById('dropdown');
const toggle = document.getElementById('dropdown-toggle');

toggle.addEventListener('click', e => {
  e.stopPropagation();
  const isOpen = !dropdown.hidden;
  dropdown.hidden = isOpen;
  toggle.setAttribute('aria-expanded', String(!isOpen));
});

document.addEventListener('click', () => {
  dropdown.hidden = true;
  toggle.setAttribute('aria-expanded', 'false');
});

dropdown.addEventListener('click', e => e.stopPropagation());

/* OS selection */

function selectOS(os) {
  selectedOS = os;
  document.getElementById('btn-label').textContent = `Download for ${OS_LABELS[os]}`;
  dropdown.hidden = true;
  toggle.setAttribute('aria-expanded', 'false');
}

document.querySelectorAll('[data-os]').forEach(btn => {
  btn.addEventListener('click', () => selectOS(btn.dataset.os));
});

/* Init */

selectOS(selectedOS);
loadRelease();
