#!/usr/bin/env node
const { spawn } = require('child_process');

// If running in CI or on Render, skip starting the dev server so build commands don't fail.
const isCI = !!(process.env.CI || process.env.RENDER || process.env.RENDER_INTERNAL || process.env.RENDER_SERVICE_ID || process.env.RENDER_COMMIT);

if (isCI) {
  console.log('Detected CI/Render environment — skipping dev server.');
  process.exit(0);
}

// Otherwise run vite normally, forwarding args and stdio.
const args = process.argv.slice(2);
const child = spawn('vite', args, { stdio: 'inherit' });
child.on('exit', code => process.exit(code));
child.on('error', err => {
  console.error('Failed to start vite:', err);
  process.exit(1);
});
