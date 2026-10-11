/*
 * Declas website settings. This is the only file you need to edit.
 *
 * Backend: create a free Supabase project, run supabase/schema.sql (repo root)
 * in its SQL editor, then paste the project URL and the "anon public" key below.
 * The anon key is meant to be public; the database only accepts calls to the
 * functions defined in schema.sql. While these two values are empty the site runs
 * in demo mode: downloads still work, but counts and sign-ups stay in this browser.
 */
window.DECLAS_CONFIG = {
  supabaseUrl: 'https://jozyrwbaixfiuigzabiz.supabase.co/rest/v1/',
  supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impvenlyd2JhaXhmaXVpZ3phYml6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTE3NTMyNTMsImV4cCI6MjEwNzMyOTI1M30.RGeai0azRCVtqairDfLc0NmV2AXoMsYb5_t0X6js-9s',

  repo: 'stangandaho/declas',
  version: '1.4.0',
  contactEmail: 'stangandaho@gmail.com',
  videoUrl: 'https://youtu.be/6sAMYketcKs',
  zenodoDoi: 'https://doi.org/10.5281/zenodo.20951592',

  // Downloads made before this site existed. Added to the live count.
  downloadBaseline: [
    { label: 'v1.4.0 (GitHub)', count: 31 },
    { label: 'v1.3.0 (GitHub)', count: 31 },
    { label: 'v1.1.0 (Zenodo)', count: 270 },
  ],

  // Used when the GitHub API cannot be reached.
  fallbackAssets: {
    windows: 'Declas_{v}.exe',
    'macos-arm': 'Declas-macOS-arm64_{v}.dmg',
    'macos-intel': 'Declas-macOS-Intel_{v}.dmg',
    linux: 'Declas-Linux_{v}.tar.gz',
  },

  // Leave a value empty and the page shows "Details coming soon" for it.
  donation: {
    bank: {
      accountName: 'GANDAHO STANISLAS MAHUSSI',
      bankName: 'Orabank Bénin',
      bankCode: 'BJ058',
      branchCode: '01015',
      accountNumber: '027971900501',
      ribKey: '06',
      iban: 'BJ66BJ0580101502797190050106',
      swift: 'ORBKBJBJ',
      bankAddress: 'Womey, Abomey-Calavi, Benin',
      bankAddressFr: 'Womey, Abomey-Calavi, Bénin',
      reference: 'Declas donation',
      referenceFr: 'Don Declas',
      ribPdf: 'assets/img/donate/orabank-rib.pdf',
    },
    piorabank: {
      qrImage: 'assets/img/donate/orabank-pi-qr.png',  // BCEAO PI instant-payment QR
      alias: '7deff80a-7877-4eda-86e5-9efa9a2ca5c6',   // PI alias, for "pay by alias" in banking apps
      link: '',             // optional payment link; the button is hidden when empty
    },
  },
};
