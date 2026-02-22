/**
 * Bundles @moq/lite (from local source) into an IIFE as window.MoqLite.
 * Run: node esbuild-moq.mjs
 *
 * Uses --alias to resolve workspace @moq/* imports to local source paths,
 * esbuild compiles TypeScript directly (no tsc needed).
 */
import { build } from "esbuild";
import { readFile } from "fs/promises";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const moqRoot = resolve(__dirname, "../../moq/js");

// Strip per-frame RX/TX console logging in ws-polyfill entirely.
// These fire on EVERY WebSocket frame (~thousands/sec) and the overhead
// of string formatting + console.debug calls starves the audio decode loop,
// causing the ring buffer to drain.
// Lifecycle warnings (handleError, handleClose, close()) are kept as-is.
const quietWsPolyfill = {
  name: "quiet-ws-polyfill",
  setup(build) {
    build.onLoad(
      { filter: /web-transport-ws[\\/]session\.js$/ },
      async (args) => {
        let contents = await readFile(args.path, "utf8");
        // Completely remove per-frame RX/TX logging (replace with void 0)
        contents = contents.replace(
          /console\.(warn|debug)\(\s*"\[ws-polyfill\] (?:RX|TX) [^]*?\);\s*/g,
          "/* ws-polyfill log stripped */ "
        );
        return { contents, loader: "js" };
      }
    );
  },
};

await build({
  entryPoints: [resolve(__dirname, "moq-bundle-entry.js")],
  bundle: true,
  format: "iife",
  globalName: "MoqLite",
  outfile: resolve(__dirname, "moq-lite-bundle.js"),
  // Resolve workspace @moq/* packages to local source
  alias: {
    "@moq/signals": resolve(moqRoot, "signals/src/index.ts"),
  },
  // Tell esbuild to find node_modules in our local dir (for dequal, async-mutex, etc.)
  nodePaths: [resolve(__dirname, "node_modules")],
  // esbuild handles TS natively
  loader: { ".ts": "ts" },
  target: "es2022",
  // Tree-shake unused exports
  treeShaking: true,
  // Replace import.meta.env for IIFE format (not available outside ESM)
  define: {
    "import.meta.env": "undefined",
  },
  // Source maps for debugging
  sourcemap: true,
  logLevel: "info",
  plugins: [quietWsPolyfill],
});
