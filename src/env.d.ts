/// <reference types="vite/client" />

// Optional: add explicit env typings here if you want autocomplete for specific vars:
 interface ImportMetaEnv {
 readonly VITE_API_BASE_URL: string;
//   // add more VITE_... vars here
 }
//
 interface ImportMeta {
   readonly env: ImportMetaEnv;
 }
