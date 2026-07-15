// One delegated tooltip surface for the whole dashboard. Existing and future
// `title` attributes remain the authoring API, but while an element is
// hovered/focused we remove the attribute synchronously (suppressing the
// OS/browser bubble) and render its text through Saklas chrome instead.

const TOOLTIP_ID = "saklas-tooltip";
const SHOW_DELAY_MS = 260;

export function installTooltipLayer(): () => void {
  const tooltip = document.createElement("div");
  tooltip.id = TOOLTIP_ID;
  tooltip.className = "saklas-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.setAttribute("aria-hidden", "true");
  document.body.append(tooltip);

  let activeEl: HTMLElement | null = null;
  let activeTitle = "";
  let previousDescribedBy: string | null = null;
  let showTimer: ReturnType<typeof setTimeout> | null = null;

  function titledAncestor(target: EventTarget | null): HTMLElement | null {
    if (!(target instanceof Element)) return null;
    return target.closest<HTMLElement>("[title]");
  }

  function place(anchor: HTMLElement): void {
    if (activeEl !== anchor) return;
    const rect = anchor.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const below = rect.top < tip.height + 16;
    const desiredX = rect.left + rect.width / 2;
    const x = Math.max(
      8 + tip.width / 2,
      Math.min(window.innerWidth - 8 - tip.width / 2, desiredX),
    );
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${below ? rect.bottom + 8 : rect.top - 8}px`;
    tooltip.classList.toggle("below", below);
  }

  function restoreActiveTitle(): void {
    if (!activeEl) return;
    // A reactive update may have authored a newer title while the tooltip was
    // open. Prefer it, then restore the latest text after leaving so the DOM
    // keeps its ordinary accessibility/source contract between interactions.
    const updated = activeEl.getAttribute("title");
    if (updated !== null) activeTitle = updated;
    activeEl.setAttribute("title", activeTitle);
    if (previousDescribedBy === null) activeEl.removeAttribute("aria-describedby");
    else activeEl.setAttribute("aria-describedby", previousDescribedBy);
  }

  function hide(): void {
    if (showTimer !== null) clearTimeout(showTimer);
    showTimer = null;
    tooltip.classList.remove("visible");
    tooltip.setAttribute("aria-hidden", "true");
    restoreActiveTitle();
    activeEl = null;
    activeTitle = "";
    previousDescribedBy = null;
  }

  function activate(anchor: HTMLElement, immediate: boolean): void {
    const authored = anchor.getAttribute("title") ?? "";
    if (!authored.trim() || activeEl === anchor) return;
    hide();
    activeEl = anchor;
    activeTitle = authored;
    previousDescribedBy = anchor.getAttribute("aria-describedby");
    const ids = new Set((previousDescribedBy ?? "").split(/\s+/).filter(Boolean));
    ids.add(TOOLTIP_ID);
    anchor.setAttribute("aria-describedby", [...ids].join(" "));
    // Removing in the pointer/focus event prevents a native tooltip from ever
    // reaching its display delay. It is restored by `hide()`.
    anchor.removeAttribute("title");
    tooltip.textContent = authored;
    const show = () => {
      if (activeEl !== anchor) return;
      tooltip.classList.add("visible");
      tooltip.setAttribute("aria-hidden", "false");
      place(anchor);
    };
    if (immediate) show();
    else showTimer = setTimeout(show, SHOW_DELAY_MS);
  }

  function onMouseOver(event: MouseEvent): void {
    const anchor = titledAncestor(event.target);
    if (anchor) activate(anchor, false);
  }

  function onMouseOut(event: MouseEvent): void {
    if (!activeEl) return;
    const related = event.relatedTarget;
    if (related instanceof Node && activeEl.contains(related)) return;
    if (document.activeElement === activeEl) return;
    hide();
  }

  function onFocusIn(event: FocusEvent): void {
    const anchor = titledAncestor(event.target);
    if (anchor) activate(anchor, true);
  }

  function onFocusOut(event: FocusEvent): void {
    if (!activeEl) return;
    const related = event.relatedTarget;
    if (related instanceof Node && activeEl.contains(related)) return;
    hide();
  }

  function onViewportChange(): void {
    if (activeEl && tooltip.classList.contains("visible")) place(activeEl);
  }

  document.addEventListener("mouseover", onMouseOver, true);
  document.addEventListener("mouseout", onMouseOut, true);
  document.addEventListener("focusin", onFocusIn, true);
  document.addEventListener("focusout", onFocusOut, true);
  window.addEventListener("resize", onViewportChange);
  window.addEventListener("scroll", onViewportChange, true);

  return () => {
    document.removeEventListener("mouseover", onMouseOver, true);
    document.removeEventListener("mouseout", onMouseOut, true);
    document.removeEventListener("focusin", onFocusIn, true);
    document.removeEventListener("focusout", onFocusOut, true);
    window.removeEventListener("resize", onViewportChange);
    window.removeEventListener("scroll", onViewportChange, true);
    hide();
    tooltip.remove();
  };
}
