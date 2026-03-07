/**
 * Bakup.ai – Frontend Application
 *
 * Key-gate logic:
 *   - Access key is validated by comparing its SHA-256 hash (via Web Crypto API)
 *     against a hardcoded expected hash. The key itself is never stored in source.
 *   - On success, a download link and checksum are revealed client-side.
 *
 * Security note:
 *   This is controlled access, not DRM. The hash constant below is visible in
 *   source. A determined person can reverse it. That is intentional and accepted —
 *   this gate is designed to slow down casual unauthorized access and maintain a
 *   feedback loop with known testers. It is not a cryptographic security boundary.
 *
 *   To rotate the key: update VALID_KEY_HASH and DOWNLOAD_CONFIG.url below.
 *   Run `node scripts/generate-key-hash.js <newkey>` to get the new hash.
 */

'use strict';

// ─── Configuration ────────────────────────────────────────────────────────────
// SHA-256 hash of the valid access key.
// Generated via: node scripts/generate-key-hash.js tango
const VALID_KEY_HASH = '7063d51d1b2da165eee042de5d33cc27281ea80e1a291488c903b7fb5fc31da7';

// Download artefact served when the key validates.
// Replace url with your real CDN path before deploying.
const DOWNLOAD_CONFIG = {
  url:      'downloads/bakup-ai-installer.exe', // relative or absolute CDN path
  checksum: '', // populated alongside VALID_KEY_HASH at deploy time
  filename: 'bakup-ai-installer.exe',
};

// ─── Utility: SHA-256 via Web Crypto ─────────────────────────────────────────
async function sha256Hex(message) {
  const encoder = new TextEncoder();
  const data     = encoder.encode(message.trim());
  const hashBuf  = await crypto.subtle.digest('SHA-256', data);
  const hashArr  = Array.from(new Uint8Array(hashBuf));
  return hashArr.map(b => b.toString(16).padStart(2, '0')).join('');
}

// Constant-time string comparison (best-effort in JS; avoids trivial timing leaks)
function safeEquals(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

// ─── Key Gate ─────────────────────────────────────────────────────────────────
const gateForm    = document.getElementById('gate-form');
const keyInput    = document.getElementById('access-key');
const gateError   = document.getElementById('gate-error');
const gateSuccess = document.getElementById('gate-success');
const downloadLink = document.getElementById('download-link');
const checksumValue = document.getElementById('checksum-value');

function showError(msg) {
  gateError.textContent = msg;
  gateError.hidden = false;
  keyInput.classList.add('error');
  keyInput.classList.remove('valid');
  keyInput.setAttribute('aria-invalid', 'true');
}

function clearError() {
  gateError.hidden = true;
  keyInput.classList.remove('error');
  keyInput.removeAttribute('aria-invalid');
}

function showSuccess() {
  gateForm.hidden = true;
  clearError();

  // Populate download link
  downloadLink.href     = DOWNLOAD_CONFIG.url;
  downloadLink.download = DOWNLOAD_CONFIG.filename;

  // Show checksum
  if (DOWNLOAD_CONFIG.checksum) {
    checksumValue.textContent = DOWNLOAD_CONFIG.checksum;
  } else {
    document.getElementById('gate-checksum').hidden = true;
  }

  gateSuccess.hidden = false;
}

if (gateForm) {
  gateForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError();

    const rawKey = keyInput.value;

    if (!rawKey.trim()) {
      showError('Access key cannot be empty.');
      return;
    }

    const submitBtn = document.getElementById('gate-submit');
    submitBtn.textContent = 'Validating…';
    submitBtn.disabled = true;

    try {
      const inputHash = await sha256Hex(rawKey);

      if (safeEquals(inputHash, VALID_KEY_HASH)) {
        keyInput.classList.add('valid');
        showSuccess();
      } else {
        showError('Invalid access key. Check your key and try again.');
        submitBtn.textContent = 'Validate';
        submitBtn.disabled = false;
      }
    } catch (err) {
      // Web Crypto not available (non-HTTPS, old browser, etc.)
      console.error('[bakup] Key validation error:', err);
      showError('Validation failed. Ensure the page is served over HTTPS.');
      submitBtn.textContent = 'Validate';
      submitBtn.disabled = false;
    }
  });

  // Clear error on new input
  keyInput.addEventListener('input', clearError);
}

// ─── Nav scroll state ─────────────────────────────────────────────────────────
const nav = document.getElementById('nav');
if (nav) {
  const onScroll = () => {
    nav.classList.toggle('scrolled', window.scrollY > 16);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll(); // apply immediately on load
}

// ─── Step cards scroll reveal ─────────────────────────────────────────────────
const steps = document.querySelectorAll('.step');
if (steps.length && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  steps.forEach(step => observer.observe(step));
} else {
  // Fallback: show all steps immediately
  steps.forEach(step => step.classList.add('visible'));
}
