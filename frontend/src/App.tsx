import { Suspense, lazy, useEffect } from "react";
import Demo from "./pages/Demo";
import { Link, usePath } from "./router";

const Brag = lazy(() => import("./pages/Brag"));
const Model = lazy(() => import("./pages/Model"));

const TITLES: Record<string, string> = {
  "/": "Music Genre CNN",
  "/brag": "Case study · Music Genre CNN",
  "/model": "Model details · Music Genre CNN",
};

const LINKS = [
  { to: "/", label: "Try it" },
  { to: "/brag", label: "Case study" },
  { to: "/model", label: "Model" },
];

export default function App() {
  const path = usePath();
  const lab = path === "/brag" || path === "/model";

  useEffect(() => {
    document.title = TITLES[path] ?? TITLES["/"];
  }, [path]);

  return (
    <div className={path === "/brag" ? "shell lab-theme cinema" : lab ? "shell lab-theme" : "shell"}>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <nav className="topnav" aria-label="Main">
        <Link to="/" className="brand">
          <svg viewBox="0 0 32 32" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" aria-hidden="true">
            <path d="M7 20v-8M12 24V8M17 20V12M22 24v-6M27 22V16" />
          </svg>
          Music Genre CNN
        </Link>
        <ul>
          {LINKS.map((link) => (
            <li key={link.to}>
              <Link to={link.to} aria-current={path === link.to ? "page" : undefined}>
                {link.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      {path === "/brag" ? (
        <Suspense fallback={<p className="route-loading">Loading case study…</p>}>
          <Brag />
        </Suspense>
      ) : path === "/model" ? (
        <Suspense fallback={<p className="route-loading">Loading model details…</p>}>
          <Model />
        </Suspense>
      ) : (
        <Demo />
      )}
    </div>
  );
}
