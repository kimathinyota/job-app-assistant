// IntelligentTextArea.jsx
// Full single-file component
// - Bootstrap dark-mode via <html data-bs-theme="dark">
// - Mobile tap fixes: pointer-tracking to distinguish taps vs scrolls
// - Scroll sensitivity relaxed (movement threshold 6px)
// - Desktop 2-pane menu, mobile anchored popup
// - Mapped items: Evidence = label, Requirement = subtitle
// - Short category names, consistent icons & spacing

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  forwardRef,
  useImperativeHandle
} from "react";
import { createPortal } from "react-dom";

import {
  Briefcase,
  GraduationCap,
  Cpu,
  Heart,
  Trophy,
  FolderGit2,
  Link as LinkIcon,
  LayoutList,
  X
} from "lucide-react";

/* ===========================
   ICON MAP (consistent sizing)
   =========================== */
const ICONS = {
  evidence: LinkIcon,
  experiences: Briefcase,
  education: GraduationCap,
  skills: Cpu,
  projects: FolderGit2,
  achievements: Trophy,
  hobbies: Heart,
  default: LayoutList
};

/* ===========================
   STYLES (single string)
   =========================== */
const STYLES = `
/* -------------------------- Editor -------------------------- */
.intelligent-editor {
  min-height: 84px;
  width: 100%;
  padding: 0.75rem;
  font-size: 0.95rem;
  line-height: 1.5;
  color: var(--bs-body-color, #1f2937);
  background-color: var(--bs-body-bg, #fff);
  border: 1px solid var(--bs-border-color, #e5e7eb);
  border-radius: 0.5rem;
  overflow-y: auto;
  white-space: pre-wrap;
  position: relative;
  transition: border-color .12s ease, box-shadow .12s ease;
  -webkit-font-smoothing: antialiased;
  -webkit-tap-highlight-color: rgba(0,0,0,0);
}

.intelligent-editor:focus {
  border-color: #86b7fe;
  box-shadow: 0 0 0 0.18rem rgba(13,110,253,0.12);
  outline: 0;
}

.intelligent-editor[data-placeholder="true"]::before {
  content: attr(data-placeholder-text);
  position: absolute;
  left: 0.75rem;
  top: 0.7rem;
  color: #9ca3af;
  opacity: 0.85;
  pointer-events: none;
  font-size: 0.95rem;
}

/* links inside text (the mention tokens) */
.intelligent-editor a {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 6px;
  border-radius: 6px;
  background-color: rgba(13,110,253,0.06);
  color: #0d6efd;
  font-weight: 600;
  text-decoration: none;
  white-space: nowrap;
  vertical-align: middle;
}

/* external markdown links look different */
.intelligent-editor a.external-link {
  background: none;
  color: #0969da;
  text-decoration: underline;
  font-weight: 500;
  padding: 0;
}

/* -------------------------- Animations -------------------------- */
@keyframes slideDownFade {
  from { opacity: 0; transform: translateY(-8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* -------------------------- Popup container -------------------------- */
.mobile-toolbar-popup {
  animation: slideDownFade 0.16s ease-out;
  background: var(--bs-body-bg, #fff);
  border-radius: 12px;
  box-shadow: 0 14px 40px -12px rgba(0,0,0,0.18);
  border: 1px solid rgba(0,0,0,0.06);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-overflow-scrolling: touch;
  -webkit-user-select: none;
  user-select: none;
  touch-action: manipulation;
}

/* Desktop panel */
.menu-desktop {
  display: flex;
  flex-direction: column;
  border-radius: 8px;
  background: var(--bs-body-bg, #fff);
}

/* -------------------------- Two-pane layout (desktop) -------------------------- */
.menu-desktop .split {
  display: flex;
  width: 100%;
  height: 100%;
}

.menu-desktop .left {
  width: 36%;
  min-width: 160px;
  max-width: 220px;
  border-right: 1px solid rgba(0,0,0,0.04);
  background: var(--bs-body-bg, #fafafa);
  overflow-y: auto;
}

.menu-desktop .right {
  width: 64%;
  overflow-y: auto;
  background: var(--bs-body-bg, #fff);
  padding: 6px 0;
}

/* left category button */
.cat-btn {
  width: 100%;
  border: 0;
  background: transparent;
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.92rem;
  text-align: left;
  cursor: pointer;
  color: var(--bs-body-color, #111827);
  touch-action: manipulation;
  -webkit-tap-highlight-color: rgba(0,0,0,0);
}

.cat-btn:hover,
.cat-btn.active {
  background: rgba(13,110,253,0.04);
}

/* small icon sizing for consistency */
.cat-btn svg, .item-icon svg {
  width: 18px;
  height: 18px;
  min-width: 18px;
  min-height: 18px;
  color: #6b7280;
  flex: 0 0 18px;
}

/* short category name styling */
.cat-label {
  font-weight: 700;
  font-size: 0.86rem;
  letter-spacing: 0.02em;
}

/* -------------------------- List items (desktop & mobile shared) -------------------------- */
.item-btn {
  width: 100%;
  border: 0;
  background: transparent;
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  text-align: left;
  cursor: pointer;
  touch-action: manipulation;
  -webkit-tap-highlight-color: rgba(0,0,0,0);
}

.item-btn:hover {
  background: rgba(0,0,0,0.02);
}

.item-body {
  display: flex;
  flex-direction: column;
  min-width: 0; /* allow truncation */
}

.item-title {
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--bs-body-color, #111827);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.item-sub {
  font-size: 0.82rem;
  color: #6b7280;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* badge on the right (category short name) */
.item-badge {
  margin-left: auto;
  font-size: 0.78rem;
  color: #0d6efd;
  background: rgba(13,110,253,0.06);
  border-radius: 8px;
  padding: 6px 8px;
  min-width: 46px;
  text-align: center;
  font-weight: 700;
}

/* make sure mobile rows can break reasonably for long content */
.mobile-toolbar-popup .item-btn .item-body .item-title,
.mobile-toolbar-popup .item-btn .item-body .item-sub {
  white-space: normal;
  word-break: break-word;
}

/* make buttons look tappable on mobile */
.mobile-toolbar-popup .item-btn {
  padding: 12px 14px;
}

/* search input */
.menu-search {
  padding: 10px;
  border-bottom: 1px solid rgba(0,0,0,0.04);
  background: transparent;
}

.menu-search input {
  width: 100%;
  padding: 8px 10px;
  font-size: 0.92rem;
}

/* dim background overlay */
.menu-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.22);
  z-index: 9998;
  touch-action: manipulation;
}

/* responsive tweaks */
@media (max-width: 767px) {
  .menu-desktop { width: auto; height: auto; }
  .menu-desktop .left { display: none; }
  .mobile-toolbar-popup { width: 92%; max-width: 720px; border-radius: 12px; }
}

/* --------------------------
   Bootstrap DARK THEME SUPPORT
   Targets structures when <html data-bs-theme="dark">
   -------------------------- */
html[data-bs-theme="dark"] .intelligent-editor {
  color: var(--bs-body-color, #e6eef8);
  background-color: var(--bs-body-bg, #0f1720);
  border-color: rgba(255,255,255,0.06);
}

html[data-bs-theme="dark"] .mobile-toolbar-popup,
html[data-bs-theme="dark"] .menu-desktop {
  background: #0b1220;
  border-color: rgba(255,255,255,0.04);
  color: #e6eef8;
  box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}

html[data-bs-theme="dark"] .menu-desktop .left {
  background: #071019;
  border-right-color: rgba(255,255,255,0.02);
}

html[data-bs-theme="dark"] .item-btn,
html[data-bs-theme="dark"] .cat-btn {
  color: #e6eef8;
  background: transparent;
  border-bottom-color: rgba(255,255,255,0.02);
}

html[data-bs-theme="dark"] .item-btn:hover,
html[data-bs-theme="dark"] .cat-btn.active {
  background: rgba(255,255,255,0.02);
}

html[data-bs-theme="dark"] .item-badge {
  color: #cfe6ff;
  background: rgba(29,78,216,0.12);
}

/* make overlay darker in dark theme */
html[data-bs-theme="dark"] .menu-overlay {
  background: rgba(0,0,0,0.6);
}
`;

/* ===========================
   Regex for parsing
   =========================== */
const REFERENCE_REGEX = /\[(.*?)]<:(.*?)><(.*?)>/g;
const MARKDOWN_LINK_REGEX = /\[([^\]]+)\]\(([^)]+)\)/g;

/* ===========================
   Utility: pointer tracking per-row
   - returns handlers and state holder
   =========================== */
function usePointerSelect(onActivate, options = {}) {
  // options: { moveThreshold: number }
  const moveThreshold = (options && options.moveThreshold) || 6;
  const metaRef = useRef({
    pointerId: null,
    startX: 0,
    startY: 0,
    isDragging: false,
    activeTarget: null
  });

  const onPointerDown = (e) => {
    // only primary button
    if (e.button && e.button !== 0) return;
    // start tracking
    const p = metaRef.current;
    p.pointerId = e.pointerId;
    p.startX = e.clientX;
    p.startY = e.clientY;
    p.isDragging = false;
    p.activeTarget = e.currentTarget;
    try {
      // capture pointer so we get move/up even if finger leaves
      e.currentTarget.setPointerCapture && e.currentTarget.setPointerCapture(e.pointerId);
    } catch (err) {
      // swallow
    }
  };

  const onPointerMove = (e) => {
    const p = metaRef.current;
    if (p.pointerId !== e.pointerId) return;
    const dx = Math.abs(e.clientX - p.startX);
    const dy = Math.abs(e.clientY - p.startY);
    if (!p.isDragging && (dx > moveThreshold || dy > moveThreshold)) {
      p.isDragging = true;
    }
  };

  const onPointerUp = (e) => {
    const p = metaRef.current;
    if (p.pointerId !== e.pointerId) return;
    try {
      p.activeTarget && p.activeTarget.releasePointerCapture && p.activeTarget.releasePointerCapture(e.pointerId);
    } catch (err) {}
    const wasDragging = p.isDragging;
    // reset
    p.pointerId = null;
    p.isDragging = false;
    p.activeTarget = null;
    // activate if not dragged
    if (!wasDragging) {
      onActivate && onActivate(e);
    }
  };

  const onPointerCancel = (e) => {
    const p = metaRef.current;
    if (p.pointerId !== e.pointerId) return;
    try {
      p.activeTarget && p.activeTarget.releasePointerCapture && p.activeTarget.releasePointerCapture(e.pointerId);
    } catch (err) {}
    p.pointerId = null;
    p.isDragging = false;
    p.activeTarget = null;
  };

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel
  };
}

/* ===========================
   HierarchicalMenu component
   =========================== */
const HierarchicalMenu = ({
  categories,
  onSelect,
  position,
  onClose,
  searchQuery = ""
}) => {
  const menuRef = useRef(null);
  const [internalQuery, setInternalQuery] = useState(searchQuery || "");
  const [activeCat, setActiveCat] = useState(
    (categories && categories.length && categories[0].id) || null
  );

  const isMobile = typeof window !== "undefined" && window.innerWidth < 768;

  // Flatten items
  const allItemsFlat = useMemo(() => {
    return (categories || []).flatMap((cat) =>
      (cat.items || []).map((item) => ({
        ...item,
        _catId: cat.id,
        _catName: cat.name,
        _catType: cat.type
      }))
    );
  }, [categories]);

  const filteredItems = useMemo(() => {
    const q = (internalQuery || "").trim().toLowerCase();
    if (!q) {
      if (!isMobile && activeCat) {
        const cat = categories.find((c) => c.id === activeCat);
        return (cat && (cat.items || []).map((i) => ({ ...i, _catId: cat.id, _catName: cat.name, _catType: cat.type }))) || [];
      }
      return allItemsFlat;
    }

    return allItemsFlat.filter((item) => {
      return (
        ((item.label || item.name) + " " + (item.subtitle || "") + " " + (item._catName || ""))
          .toLowerCase()
          .indexOf(q) !== -1
      );
    });
  }, [internalQuery, allItemsFlat, isMobile, activeCat, categories]);

  // close on outside pointerdown (fast)
  useEffect(() => {
    const handler = (e) => {
      // if click/tap occurs outside the menu, close
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        onClose();
      }
    };
    document.addEventListener("pointerdown", handler, { passive: true });
    return () => document.removeEventListener("pointerdown", handler);
  }, [onClose]);

  // Desktop style
  const desktopStyle = useMemo(() => {
    if (isMobile) return {};
    const W = 500;
    const left = Math.min(Math.max(8, position.left - 250), Math.max(8, window.innerWidth - W - 8));
    const top = Math.min(Math.max(8, position.top + 8), Math.max(8, window.innerHeight - 360));
    return {
      position: "fixed",
      top,
      left,
      width: W,
      height: 340,
      zIndex: 11020,
      display: "flex",
      flexDirection: "column"
    };
  }, [isMobile, position]);

  // Mobile anchored style
  const mobileStyle = useMemo(() => {
    if (!isMobile) return {};
    const margin = 8;
    const maxWidth = Math.min(window.innerWidth * 0.82, 760);
    const maxHeight = Math.min(window.innerHeight * 0.6, 520);
    let top;
    const spaceAbove = position.top;
    const spaceBelow = window.innerHeight - position.top;
    if (spaceBelow > maxHeight + 20 || spaceBelow >= spaceAbove) {
      top = position.top + margin;
    } else {
      top = Math.max(margin, position.top - maxHeight - margin);
    }
    let left = position.left + margin;
    if (left + maxWidth + margin > window.innerWidth) {
      left = Math.max(margin, window.innerWidth - maxWidth - margin);
    }
    return {
      position: "fixed",
      top,
      left,
      width: maxWidth,
      maxHeight,
      zIndex: 11030,
      overflowY: "auto"
    };
  }, [isMobile, position]);

  // ensure activeCat set
  useEffect(() => {
    if (!activeCat && categories && categories.length) setActiveCat(categories[0].id);
  }, [categories, activeCat]);

  const containerClass = isMobile ? "mobile-toolbar-popup" : "menu-desktop";

  // generic handler for selection that parent expects
  const doSelect = (item, type) => {
    try {
      onSelect && onSelect(item, type);
    } catch (err) {
      console.warn("onSelect error", err);
    } finally {
      onClose && onClose();
    }
  };

  // pointer handlers per item using hook
  const makePointerHandlersFor = (item, type) =>
    useMemo(() => {
      const handlers = usePointerSelect(() => {
        doSelect(item, type);
      }, { moveThreshold: 6 });
      return handlers;
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [item, type]);

  return createPortal(
    <>
      {/* overlay behind menu; close when tapped */}
      <div
        className="menu-overlay"
        style={{ zIndex: isMobile ? 11010 : 10500 }}
        onPointerDown={(e) => {
          // close overlay on any pointerdown
          e.stopPropagation();
          onClose();
        }}
      />

      <div
        ref={menuRef}
        className={containerClass}
        style={isMobile ? mobileStyle : desktopStyle}
        // don't prevent default here — we let children handle pointer capture
        onPointerDown={(e) => {
          // stop propagation so outer document listener doesn't immediately close
          e.stopPropagation();
        }}
      >
        {/* header + search */}
        <div className="menu-search" style={{ zIndex: 11040 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {isMobile ? (
              <>
                <strong style={{ fontSize: 14, color: "var(--bs-body-color, #374151)" }}>Link Evidence</strong>
                <div style={{ flex: 1 }} />
                <button
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    onClose();
                  }}
                  className="btn btn-sm"
                  aria-label="close"
                >
                  <X size={16} />
                </button>
              </>
            ) : (
              <strong style={{ fontSize: 14, color: "var(--bs-body-color, #374151)" }}>Insert Link</strong>
            )}
          </div>

          <div style={{ marginTop: 8 }}>
            <input
              aria-label="Search"
              placeholder="Search..."
              value={internalQuery}
              onChange={(e) => setInternalQuery(e.target.value)}
              className="form-control"
              style={{ padding: "8px 10px", fontSize: 14 }}
              autoFocus
              onPointerDown={(e) => e.stopPropagation()}
            />
          </div>
        </div>

        {/* mobile unified list */}
        {isMobile ? (
          <div style={{ flex: 1, overflowY: "auto" }}>
            {filteredItems.length === 0 ? (
              <div style={{ padding: 16, color: "#6b7280", fontStyle: "italic" }}>No matches found</div>
            ) : (
              filteredItems.map((item) => {
                const Icon = ICONS[item._catType] || ICONS.default;
                // do not create new handlers inside render synchronously — wrap in closure
                const handlers = (() => {
                  // local pointer-tracking instance (per rendered item)
                  const metaRef = {
                    pointerId: null,
                    startX: 0,
                    startY: 0,
                    isDragging: false,
                    activeTarget: null
                  };

                  const onPointerDown = (e) => {
                    if (e.button && e.button !== 0) return;
                    metaRef.pointerId = e.pointerId;
                    metaRef.startX = e.clientX;
                    metaRef.startY = e.clientY;
                    metaRef.isDragging = false;
                    metaRef.activeTarget = e.currentTarget;
                    try {
                      e.currentTarget.setPointerCapture && e.currentTarget.setPointerCapture(e.pointerId);
                    } catch (err) {}
                  };

                  const onPointerMove = (e) => {
                    if (metaRef.pointerId !== e.pointerId) return;
                    const dx = Math.abs(e.clientX - metaRef.startX);
                    const dy = Math.abs(e.clientY - metaRef.startY);
                    if (!metaRef.isDragging && (dx > 6 || dy > 6)) metaRef.isDragging = true;
                  };

                  const onPointerUp = (e) => {
                    if (metaRef.pointerId !== e.pointerId) return;
                    try {
                      metaRef.activeTarget && metaRef.activeTarget.releasePointerCapture && metaRef.activeTarget.releasePointerCapture(e.pointerId);
                    } catch (err) {}
                    const wasDragging = metaRef.isDragging;
                    metaRef.pointerId = null;
                    metaRef.isDragging = false;
                    metaRef.activeTarget = null;
                    if (!wasDragging) {
                      doSelect(item, item._catType);
                    }
                  };

                  const onPointerCancel = (e) => {
                    if (metaRef.pointerId !== e.pointerId) return;
                    try {
                      metaRef.activeTarget && metaRef.activeTarget.releasePointerCapture && metaRef.activeTarget.releasePointerCapture(e.pointerId);
                    } catch (err) {}
                    metaRef.pointerId = null;
                    metaRef.isDragging = false;
                    metaRef.activeTarget = null;
                  };

                  return {
                    onPointerDown,
                    onPointerMove,
                    onPointerUp,
                    onPointerCancel
                  };
                })();

                return (
                  <button
                    key={item.id}
                    className="item-btn"
                    type="button"
                    onPointerDown={handlers.onPointerDown}
                    onPointerMove={handlers.onPointerMove}
                    onPointerUp={handlers.onPointerUp}
                    onPointerCancel={handlers.onPointerCancel}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        doSelect(item, item._catType);
                      }
                    }}
                  >
                    <span className="item-icon" style={{ display: "inline-flex", alignItems: "center" }}>
                      {React.createElement(Icon, { size: 18 })}
                    </span>

                    <div className="item-body" style={{ marginLeft: 4 }}>
                      <div className="item-title">{item.label || item.name}</div>
                      {item.subtitle ? <div className="item-sub">{item.subtitle}</div> : null}
                    </div>

                    <div className="item-badge" aria-hidden>{item._catName}</div>
                  </button>
                );
              })
            )}
          </div>
        ) : (
          /* desktop 2-pane: left categories, right items */
          <div className="split" style={{ flex: 1 }}>
            <div className="left">
              {(categories || []).map((cat) => {
                const Icon = ICONS[cat.type] || ICONS.default;
                const isActive = cat.id === activeCat;
                return (
                  <button
                    key={cat.id}
                    className={`cat-btn ${isActive ? "active" : ""}`}
                    type="button"
                    onPointerDown={(e) => {
                      // short, immediate
                      e.stopPropagation();
                      setActiveCat(cat.id);
                      setInternalQuery("");
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setActiveCat(cat.id);
                        setInternalQuery("");
                      }
                    }}
                  >
                    <span className="item-icon">{React.createElement(Icon, { size: 18 })}</span>
                    <span className="cat-label">{cat.name}</span>
                  </button>
                );
              })}
            </div>

            <div className="right">
              {filteredItems.length === 0 ? (
                <div style={{ padding: 16, color: "#6b7280", fontStyle: "italic" }}>No matches in this view</div>
              ) : (
                filteredItems.map((item) => {
                  // create pointer handlers for desktop items too
                  const metaRef = {
                    pointerId: null,
                    startX: 0,
                    startY: 0,
                    isDragging: false,
                    activeTarget: null
                  };

                  const onPointerDown = (e) => {
                    if (e.button && e.button !== 0) return;
                    metaRef.pointerId = e.pointerId;
                    metaRef.startX = e.clientX;
                    metaRef.startY = e.clientY;
                    metaRef.isDragging = false;
                    metaRef.activeTarget = e.currentTarget;
                    try {
                      e.currentTarget.setPointerCapture && e.currentTarget.setPointerCapture(e.pointerId);
                    } catch (err) {}
                    e.stopPropagation();
                  };

                  const onPointerMove = (e) => {
                    if (metaRef.pointerId !== e.pointerId) return;
                    const dx = Math.abs(e.clientX - metaRef.startX);
                    const dy = Math.abs(e.clientY - metaRef.startY);
                    if (!metaRef.isDragging && (dx > 6 || dy > 6)) metaRef.isDragging = true;
                  };

                  const onPointerUp = (e) => {
                    if (metaRef.pointerId !== e.pointerId) return;
                    try {
                      metaRef.activeTarget && metaRef.activeTarget.releasePointerCapture && metaRef.activeTarget.releasePointerCapture(e.pointerId);
                    } catch (err) {}
                    const wasDragging = metaRef.isDragging;
                    metaRef.pointerId = null;
                    metaRef.isDragging = false;
                    metaRef.activeTarget = null;
                    if (!wasDragging) doSelect(item, item._catType);
                  };

                  const onPointerCancel = (e) => {
                    if (metaRef.pointerId !== e.pointerId) return;
                    try {
                      metaRef.activeTarget && metaRef.activeTarget.releasePointerCapture && metaRef.activeTarget.releasePointerCapture(e.pointerId);
                    } catch (err) {}
                    metaRef.pointerId = null;
                    metaRef.isDragging = false;
                    metaRef.activeTarget = null;
                  };

                  return (
                    <button
                      key={item.id}
                      className="item-btn"
                      type="button"
                      onPointerDown={(e) => {
                        onPointerDown(e);
                      }}
                      onPointerMove={(e) => onPointerMove(e)}
                      onPointerUp={(e) => onPointerUp(e)}
                      onPointerCancel={(e) => onPointerCancel(e)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          doSelect(item, item._catType);
                        }
                      }}
                    >
                      <div style={{ display: "flex", gap: 12, alignItems: "center", minWidth: 0 }}>
                        <span className="item-icon" aria-hidden>
                          {React.createElement(ICONS[item._catType] || ICONS.default, { size: 18 })}
                        </span>
                        <div className="item-body">
                          <div className="item-title">{item.label || item.name}</div>
                          {item.subtitle ? <div className="item-sub">{item.subtitle}</div> : null}
                        </div>
                        <div className="item-badge" aria-hidden>{item._catName}</div>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
    </>,
    document.body
  );
};

/* ===========================
   IntelligentTextArea main component
   =========================== */
const IntelligentTextArea = forwardRef(
  (
    {
      initialValue,
      onSave,
      cv,
      extraSuggestions = [],
      onMention,
      placeholder = "Type @ to link evidence..."
    },
    ref
  ) => {
    const editorRef = useRef(null);
    const [menuState, setMenuState] = useState({
      open: false,
      position: { top: 0, left: 0 },
      mode: "cursor", // 'cursor' or 'external'
      query: ""
    });

    const mentionMetaRef = useRef(null);

    /* -------------------------
       Build categories (short names)
       ------------------------- */
    const categories = useMemo(() => {
      const cats = [];
      // Mapped (short name MAP) - Requirement as subtitle
      if ((extraSuggestions || []).length > 0) {
        cats.push({
          id: "mapped",
          type: "evidence",
          name: "MAP",
          items: (extraSuggestions || []).map((s, idx) => ({
            ...s,
            // Present evidence first and requirement underneath
            label: s.evidence || s.label || `Evidence ${idx + 1}`,
            subtitle: s.requirement || s.subtitle || ""
          }))
        });
      }

      // helper
      const toItems = (list, labelKey, subKey) =>
        (list || []).map((i) => ({
          ...i,
          label: i[labelKey],
          subtitle: i[subKey]
        }));

      if (cv) {
        cats.push({
          id: "exp",
          type: "experiences",
          name: "EXP",
          items: toItems(cv.experiences, "title", "company")
        });
        cats.push({
          id: "proj",
          type: "projects",
          name: "PRO",
          items: toItems(cv.projects, "title", "description")
        });
        cats.push({
          id: "skill",
          type: "skills",
          name: "SK",
          items: toItems(cv.skills, "name", "category")
        });
        cats.push({
          id: "ach",
          type: "achievements",
          name: "ACH",
          items: toItems(cv.achievements, "text", "context")
        });
        cats.push({
          id: "edu",
          type: "education",
          name: "EDU",
          items: toItems(cv.education, "degree", "institution")
        });
        cats.push({
          id: "hob",
          type: "hobbies",
          name: "HOB",
          items: toItems(cv.hobbies, "name", "")
        });
      }

      return cats.filter((c) => (c.items || []).length > 0);
    }, [cv, extraSuggestions]);

    /* -------------------------
       Raw <-> HTML parsing
       ------------------------- */
    const parseRawToHtml = (text) =>
      text
        ? text
            .replace(REFERENCE_REGEX, (m, t, i, n) => {
              // t = type, i = id, n = visible name
              return `<a href="#" data-type="${t}" data-id="${i}" contenteditable="false">${n}</a>`;
            })
            .replace(MARKDOWN_LINK_REGEX, (m, l, u) => {
              return `<a href="${u}" target="_blank" rel="noopener noreferrer" class="external-link" contenteditable="false">${l}</a>`;
            })
        : "";

    const parseHtmlToRaw = (el) => {
      let raw = "";
      el.childNodes.forEach((node) => {
        if (node.nodeType === 3) {
          raw += node.textContent;
        } else if (node.tagName === "A") {
          if (node.dataset && node.dataset.type) {
            raw += `[${node.dataset.type}]<:${node.dataset.id}><${node.textContent}>`;
          } else {
            const href = node.getAttribute("href");
            if (href) raw += `[${node.textContent}](${href})`;
            else raw += node.textContent;
          }
        } else if (node.tagName === "BR") {
          raw += "\n";
        } else {
          raw += parseHtmlToRaw(node);
        }
      });
      return raw;
    };

    /* -------------------------
       Caret positioning helper
       ------------------------- */
    const getCaretCoordinates = () => {
      const sel = window.getSelection();
      if (!sel || !sel.rangeCount) return { top: 0, left: 0 };
      const range = sel.getRangeAt(0).cloneRange();
      range.collapse(true);
      const rects = range.getClientRects();
      if (rects && rects.length) {
        const rect = rects[0];
        return { top: rect.bottom, left: rect.left };
      }
      return { top: 0, left: 0 };
    };

    /* -------------------------
       Click handler (external link opening)
       ------------------------- */
    const handleClick = (e) => {
      const t = e.target;
      if (t && t.tagName === "A" && t.classList.contains("external-link")) {
        const href = t.getAttribute("href");
        if (href) {
          e.preventDefault();
          window.open(href, "_blank", "noopener,noreferrer");
        }
      }
    };

    /* -------------------------
       Input handling (detect @)
       ------------------------- */
    const handleInput = () => {
      const sel = window.getSelection();
      if (!sel || !sel.rangeCount) return;
      const node = sel.anchorNode;
      if (!node || node.nodeType !== 3) return;

      const text = node.textContent || "";
      const offset = sel.anchorOffset;
      const before = text.slice(0, offset);

      const idx = before.lastIndexOf("@");
      if (idx !== -1) {
        const query = before.slice(idx + 1);
        mentionMetaRef.current = { node, offset: idx };
        setMenuState({
          open: true,
          position: getCaretCoordinates(),
          mode: "cursor",
          query
        });
      } else {
        // hide
        setMenuState((s) => ({ ...s, open: false, query: "" }));
      }
      if (editorRef.current) {
        editorRef.current.dataset.placeholder = editorRef.current.textContent.trim().length === 0;
      }
    };

    /* -------------------------
       Insert selected item
       ------------------------- */
    const insertItem = (item, type) => {
      // if menu opened via external API (mode === 'external'), just notify
      if (menuState.mode === "external") {
        setMenuState((s) => ({ ...s, open: false, query: "" }));
        if (onMention) onMention(item, type);
        return;
      }

      const sel = window.getSelection();
      if (!sel || !sel.rangeCount) return;
      const meta = mentionMetaRef.current;
      if (!meta || !meta.node) return;

      const range = document.createRange();
      const { node, offset } = meta;
      const currentCursor = sel.anchorOffset;

      // remove the @... text from the node
      range.setStart(node, offset);
      range.setEnd(node, currentCursor);
      range.deleteContents();

      // create anchor node
      const a = document.createElement("a");
      a.href = "#";
      a.contentEditable = false;
      a.dataset.type = type;
      a.dataset.id = item.id;
      a.textContent = item.label || item.name || "";

      range.insertNode(a);
      // insert normal space after anchor so caret moves cleanly
      const spacer = document.createTextNode("\u00A0");
      a.after(spacer);

      // move caret after space
      range.setStartAfter(spacer);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);

      setMenuState({ open: false, position: { top: 0, left: 0 }, mode: "cursor", query: "" });
      if (editorRef.current) editorRef.current.focus();

      if (onMention) onMention(item, type);
    };

    /* -------------------------
       Initialize editor content
       ------------------------- */
    useEffect(() => {
      if (!editorRef.current) return;
      editorRef.current.innerHTML = parseRawToHtml(initialValue || "");
      editorRef.current.dataset.placeholder = !initialValue;
    }, [initialValue]);

    /* -------------------------
       Blur saves content
       ------------------------- */
    const handleBlur = () => {
      if (!editorRef.current) return;
      const raw = parseHtmlToRaw(editorRef.current);
      if (raw !== initialValue && onSave) onSave(raw);
    };

    /* -------------------------
       Expose public API
       ------------------------- */
    useImperativeHandle(ref, () => ({
      openMenu: (rect) => {
        if (!rect) return;
        setMenuState({
          open: true,
          position: { top: rect.bottom, left: rect.left },
          mode: "external",
          query: ""
        });
      },
      triggerSearch: () => {
        if (editorRef.current) editorRef.current.focus();
      }
    }));

    return (
      <div className="intelligent-textarea-wrapper">
        <style>{STYLES}</style>

        <div
          ref={editorRef}
          className="intelligent-editor"
          contentEditable
          onInput={handleInput}
          onBlur={handleBlur}
          onClick={handleClick}
          data-placeholder-text={placeholder}
          spellCheck="false"
        />

        {menuState.open && (
          <HierarchicalMenu
            categories={categories}
            position={menuState.position}
            onSelect={(item, type) => insertItem(item, type)}
            onClose={() => setMenuState((s) => ({ ...s, open: false }))}
            searchQuery={menuState.query}
          />
        )}
      </div>
    );
  }
);

export default IntelligentTextArea;
