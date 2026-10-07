import { useEffect, useRef } from "react";
import { useActiveChapter } from "./engine";

export interface ChapterInfo {
  id: string;
  label: string;
}

/** Whole-film progress line plus a chapter index (desktop). Both are plain links/CSS: no state per scroll tick. */
export function Hud({ chapters }: { chapters: readonly ChapterInfo[] }) {
  const bar = useRef<HTMLDivElement>(null);
  const active = useActiveChapter();

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const total = document.documentElement.scrollHeight - window.innerHeight;
      bar.current?.style.setProperty("--film", total > 0 ? String(Math.min(1, window.scrollY / total)) : "0");
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <>
      <div ref={bar} className="f-bar" aria-hidden="true" />
      <nav className="f-hud" aria-label="Chapters">
        <ol>
          {chapters.map((chapter) => (
            <li key={chapter.id}>
              <a href={`#ch-${chapter.id}`} aria-current={active === chapter.id ? "true" : undefined}>
                <span className="l">{chapter.label}</span>
                <i />
              </a>
            </li>
          ))}
        </ol>
      </nav>
    </>
  );
}
