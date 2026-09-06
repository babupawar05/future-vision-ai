const CACHE_NAME = 'future-vision-v1';
const ASSETS = ['/', '/index.html', '/manifest.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener('fetch', (e) => {
  if (e.request.url.includes('/api/')) {
    return fetch(e.request);
  }
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request))
  );
});