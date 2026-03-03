/**
 * Usage: node scripts/generate-key-hash.js <access-key>
 *
 * Outputs the SHA-256 hex hash of the given key.
 * Copy the result into VALID_KEY_HASH in app.js.
 *
 * This script has no dependencies. Node.js 15+ required (built-in crypto).
 */
const { createHash } = require('crypto');

const key = process.argv[2];

if (!key) {
  console.error('Usage: node scripts/generate-key-hash.js <access-key>');
  process.exit(1);
}

const hash = createHash('sha256').update(key.trim(), 'utf8').digest('hex');

console.log('\nKey     :', key);
console.log('SHA-256 :', hash);
console.log('\nPaste into app.js → VALID_KEY_HASH:\n');
console.log(`const VALID_KEY_HASH = '${hash}';`);
