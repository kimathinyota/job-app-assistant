// frontend/src/components/applications/RichTextEditor.jsx
import React, { useState, useEffect, useRef, useMemo, createContext, useContext } from 'react';
import { createPortal } from 'react-dom';
import { useEditor, EditorContent, ReactRenderer, NodeViewWrapper, NodeViewContent, ReactNodeViewRenderer, mergeAttributes } from '@tiptap/react';
import { Node, Mark, InputRule } from '@tiptap/core'; 
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import Mention from '@tiptap/extension-mention';
import Link from '@tiptap/extension-link';
import { 
    Bold, Italic, List, ListOrdered, Quote, 
    Briefcase, GraduationCap, Cpu, Heart, Trophy, 
    Lightbulb, ChevronDown, Type, Sparkles, Info, Ghost, CheckCircle2, Trash2, Eye, EyeOff,
    SquareDashedBottom, Link2, X, ChevronRight, LayoutTemplate, FileText, Copy
} from 'lucide-react';
import tippy from 'tippy.js';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/shift-away.css';
import 'tippy.js/themes/light-border.css';

// --- CONTEXT ---
const EditorActionContext = createContext({
    onPreview: () => {},
    cvCategories: {},
    linkableItems: [] 
});

// --- PREVIEW SIDE PANEL (Modern Offcanvas) ---
const PreviewSidePanel = ({ isOpen, onClose, content }) => {
    const showClass = isOpen ? 'show' : '';
    const visibility = isOpen ? 'visible' : 'hidden';

    const handleCopy = async () => {
        if (!content) return;
        try {
            const blobHtml = new Blob([content], { type: "text/html" });
            const blobText = new Blob([content.replace(/<[^>]+>/g, '')], { type: "text/plain" });
            const data = [new ClipboardItem({ "text/html": blobHtml, "text/plain": blobText })];
            await navigator.clipboard.write(data);
            alert("Copied formatted text! You can now paste into Word or Google Docs.");
        } catch (err) {
            console.error("Copy failed: ", err);
            alert("Clipboard access denied. Please manually copy the text.");
        }
    };

    return createPortal(
        <>
            {/* Backdrop */}
            {isOpen && (
                <div 
                    className="offcanvas-backdrop fade show" 
                    onClick={onClose}
                    style={{ zIndex: 1045 }} 
                ></div>
            )}

            {/* Side Panel */}
            <div 
                className={`offcanvas offcanvas-end ${showClass} shadow-lg border-start d-flex flex-column`} 
                tabIndex="-1" 
                style={{ visibility, width: '600px', zIndex: 1050 }}
            >
                <div className="offcanvas-header border-bottom p-4 bg-light">
                    <div className="d-flex align-items-center gap-3">
                        <div className="bg-primary text-white p-2 rounded-circle d-flex align-items-center justify-content-center" style={{width: 40, height: 40}}>
                            <FileText size={20} />
                        </div>
                        <div>
                            <h5 className="offcanvas-title fw-bold text-dark m-0" style={{lineHeight: 1.2}}>Document Preview</h5>
                            <span className="text-muted small">Clean version (Ghost text removed)</span>
                        </div>
                    </div>
                    <button 
                        type="button" 
                        className="btn btn-icon btn-sm btn-light rounded-circle" 
                        onClick={onClose}
                    >
                        <X size={20} />
                    </button>
                </div>
                
                <div className="offcanvas-body p-0 bg-light-subtle custom-scroll">
                    {/* Document Paper View */}
                    <div className="p-4 p-md-5 min-h-100">
                        {content ? (
                            <div 
                                className="bg-white shadow-sm p-5 mx-auto border" 
                                style={{
                                    fontFamily: '"Times New Roman", Times, serif', 
                                    fontSize: '12pt', 
                                    lineHeight: '1.6', 
                                    color: '#000',
                                    minHeight: '800px',
                                    maxWidth: '100%'
                                }}
                            >
                                <div dangerouslySetInnerHTML={{ __html: content }} />
                            </div>
                        ) : (
                            <div className="h-100 d-flex flex-column align-items-center justify-content-center text-muted opacity-50">
                                <Ghost size={48} className="mb-3" />
                                <p>No visible content yet.</p>
                                <span className="small">Start typing or unhide sections to see them here.</span>
                            </div>
                        )}
                    </div>
                </div>
                
                <div className="offcanvas-footer p-3 border-top bg-white d-flex gap-2 justify-content-end">
                    <button className="btn btn-outline-secondary" onClick={onClose}>
                        Close
                    </button>
                    <button className="btn btn-primary d-flex align-items-center gap-2" onClick={handleCopy} disabled={!content}>
                        <Copy size={16}/> Copy to Clipboard
                    </button>
                </div>
            </div>
        </>,
        document.body
    );
};

// --- PORTAL DROPDOWN ---
const PortalDropdown = ({ isOpen, onClose, rect, children }) => {
    if (!isOpen || !rect) return null;
    const style = {
        position: 'fixed',
        top: `${rect.bottom + 8}px`,
        left: `${rect.left}px`,
        zIndex: 9999,
        minWidth: '240px',
        maxWidth: '300px',
        maxHeight: '300px',
        overflowY: 'auto'
    };
    return createPortal(
        <>
            <div style={{position: 'fixed', inset: 0, zIndex: 9998}} onClick={onClose} />
            <div className="bg-white border rounded-3 shadow-lg animate-fade-in" style={style}>{children}</div>
        </>,
        document.body
    );
};

// --- HIERARCHICAL LINK MENU ---
const HierarchicalLinkMenu = ({ cvCategories, linkableItems, onSelect, onClose }) => {
    const [activeCategory, setActiveCategory] = useState(null);

    const categories = useMemo(() => {
        return [
            { id: 'reqs', label: 'Requirements', icon: Lightbulb, items: linkableItems.filter(i => i.type === 'requirement') },
            { id: 'exp', label: 'Experience', icon: Briefcase, items: cvCategories.Experience || [] },
            { id: 'edu', label: 'Education', icon: GraduationCap, items: cvCategories.Education || [] },
            { id: 'skills', label: 'Skills', icon: Cpu, items: cvCategories.Skills || [] },
            { id: 'ach', label: 'Achievements', icon: Trophy, items: cvCategories.Achievements || [] },
            { id: 'hobby', label: 'Hobbies', icon: Heart, items: cvCategories.Hobbies || [] },
        ].filter(c => c.items && c.items.length > 0);
    }, [cvCategories, linkableItems]);

    return (
        <div className="bg-white shadow-lg border rounded-3 overflow-hidden d-flex" style={{height: '250px', width: '400px'}}>
            <div className="w-40 border-end bg-light overflow-auto custom-scroll">
                {categories.map(cat => (
                    <button
                        key={cat.id}
                        className={`w-100 btn btn-sm text-start d-flex align-items-center justify-content-between px-3 py-2 border-0 rounded-0 ${activeCategory === cat.id ? 'bg-white text-primary fw-bold shadow-inset' : 'text-muted hover-bg-white'}`}
                        onMouseEnter={() => setActiveCategory(cat.id)}
                        onMouseDown={(e) => e.preventDefault()}
                    >
                        <div className="d-flex align-items-center gap-2">
                            <cat.icon size={14} /> {cat.label}
                        </div>
                        <ChevronRight size={12} />
                    </button>
                ))}
            </div>
            <div className="w-60 overflow-auto custom-scroll bg-white">
                {!activeCategory ? (
                    <div className="h-100 d-flex align-items-center justify-content-center text-muted small fst-italic p-3 text-center">
                        Hover a category to see items...
                    </div>
                ) : (
                    categories.find(c => c.id === activeCategory)?.items.map(item => (
                        <button
                            key={item.id}
                            className="w-100 btn btn-sm text-start text-truncate px-3 py-2 border-bottom border-light hover-bg-primary-subtle small"
                            onClick={() => { onSelect(item); onClose(); }}
                            onMouseDown={(e) => e.preventDefault()}
                            title={item.label}
                        >
                            {item.label}
                        </button>
                    ))
                )}
            </div>
        </div>
    );
};

// ============================================================================
// 1. CODEX EXTENSIONS
// ============================================================================

// --- GHOST TEXT MARK (Highlighter) ---
const GhostText = Mark.create({
    name: 'ghostText',
    addAttributes() { return { class: { default: 'ghost-text-mark' } } },
    parseHTML() { return [{ tag: 'span.ghost-text-mark' }] },
    renderHTML({ HTMLAttributes }) { return ['span', mergeAttributes(HTMLAttributes, { class: 'ghost-text-mark' }), 0] },
});

// --- GHOST SECTION NODE ---
const GhostSection = Node.create({
    name: 'ghostSection',
    group: 'block',
    // Allows nesting of blocks and sections
    content: '(paragraph | blockquote | bulletList | orderedList | ghostSection)+', 
    defining: true, 
    
    addAttributes() {
        return {
            label: { default: 'New Strategy Section' },
            linkedId: { default: null },
            linkedType: { default: null },
            isVisible: { default: false }, 
        };
    },

    parseHTML() { return [{ tag: 'div[data-type="ghost-section"]' }]; },

    renderHTML({ HTMLAttributes }) {
        return ['div', mergeAttributes(HTMLAttributes, { 'data-type': 'ghost-section', class: 'ghost-section-block' }), 0];
    },

    addNodeView() { return ReactNodeViewRenderer(GhostSectionComponent); },

    addKeyboardShortcuts() {
        return {
            Backspace: () => {
                const { state } = this.editor;
                const { selection } = state;
                const { empty, $anchor } = selection;
                
                if (!empty) return false;

                let ghostSectionNode = null;
                for (let d = $anchor.depth; d > 0; d--) {
                    const node = $anchor.node(d);
                    if (node.type.name === this.name) {
                        ghostSectionNode = node;
                        break;
                    }
                }

                if (!ghostSectionNode) return false;

                const parent = $anchor.parent;
                const isFirstChild = ghostSectionNode.firstChild === parent;
                const isAtStartOfParent = $anchor.parentOffset === 0;

                if (isFirstChild && isAtStartOfParent) {
                    if (ghostSectionNode.textContent.trim().length > 0) {
                        return true; 
                    }
                    return false; 
                }
                return false;
            },
            // Enter shortcut removed to allow normal Tiptap newline insertion
        };
    },

    addInputRules() {
        return [
            new InputRule({
                find: /^>>>\s$/, 
                handler: ({ state, range }) => {
                    const { tr } = state;
                    tr.replaceWith(range.from - 1, range.to, this.type.create({}, state.schema.nodes.paragraph.create()));
                },
            }),
        ];
    },
});

// --- GHOST SECTION COMPONENT ---
const GhostSectionComponent = (props) => {
    const { node, updateAttributes, deleteNode } = props;
    const { linkableItems, cvCategories } = useContext(EditorActionContext);
    const [isEditing, setIsEditing] = useState(false);
    const [isLinking, setIsLinking] = useState(false);
    const inputRef = useRef(null);

    useEffect(() => { 
        if (isEditing && inputRef.current) {
            inputRef.current.focus();
            const len = inputRef.current.value.length;
            inputRef.current.setSelectionRange(len, len);
        }
    }, [isEditing]);

    const handleInput = (e) => {
        const val = e.target.value;
        if (val.endsWith('@')) {
            setIsLinking(true); 
            updateAttributes({ label: val.slice(0, -1) }); 
        } else {
            updateAttributes({ label: val });
        }
    };

    const handleLink = (item) => {
        updateAttributes({ linkedId: item.id, label: item.label, linkedType: item.type });
        setIsLinking(false);
        setIsEditing(false);
    };

    const toggleVisibility = () => {
        updateAttributes({ isVisible: !node.attrs.isVisible });
    };

    return (
        <NodeViewWrapper className="ghost-section-wrapper my-4">
            <div className={`ghost-header d-flex align-items-center gap-2 p-2 rounded-top border border-dashed 
                ${node.attrs.linkedId ? 'border-primary bg-primary-subtle text-primary' : 'border-purple bg-purple-subtle text-purple'}
                ${node.attrs.isVisible ? 'border-bottom-0 border-solid shadow-sm' : ''} 
            `}>
                <button 
                    className="btn btn-sm btn-link p-0 text-inherit hover-opacity-100 opacity-75" 
                    onClick={toggleVisibility}
                    title={node.attrs.isVisible ? "Header is VISIBLE in preview" : "Header is HIDDEN in preview (Structure only)"}
                >
                    {node.attrs.isVisible ? <Eye size={14}/> : <EyeOff size={14} />}
                </button>

                {node.attrs.linkedId ? <Link2 size={14}/> : <SquareDashedBottom size={14}/>}

                {isEditing ? (
                    <div className="flex-grow-1 position-relative">
                         <input
                            ref={inputRef}
                            className="form-control form-control-sm border-0 bg-transparent p-0 fw-bold text-inherit w-100 shadow-none"
                            value={node.attrs.label}
                            onChange={handleInput}
                            onBlur={() => setTimeout(() => { if(!isLinking) setIsEditing(false); }, 200)}
                            onKeyDown={(e) => e.key === 'Enter' && setIsEditing(false)}
                        />
                        {isLinking && (
                            <div className="position-absolute top-100 start-0 mt-1" style={{zIndex: 200}}>
                                <HierarchicalLinkMenu cvCategories={cvCategories} linkableItems={linkableItems} onSelect={handleLink} onClose={() => setIsLinking(false)} />
                            </div>
                        )}
                    </div>
                ) : (
                    <span 
                        className="fw-bold small flex-grow-1 cursor-text text-truncate" 
                        onClick={() => setIsEditing(true)}
                        title="Click to rename section or type '@' to link"
                    >
                        {node.attrs.label}
                    </span>
                )}

                <div className="d-flex align-items-center gap-1">
                    {node.attrs.linkedId ? (
                        <button className="btn btn-sm btn-link p-0 opacity-50 hover-opacity-100 text-inherit" onClick={() => updateAttributes({ linkedId: null, linkedType: null, label: 'New Strategy Section' })} title="Unlink"><X size={14}/></button>
                    ) : (
                        !isEditing && <button className="btn btn-sm btn-link p-0 opacity-50 hover-opacity-100 text-inherit fw-bold small" onClick={() => { setIsEditing(true); setIsLinking(true); }}>Link</button>
                    )}
                    <div className="vr mx-1 opacity-25 bg-current"></div>
                    <button className="btn btn-sm btn-link p-0 opacity-50 hover-opacity-100 text-inherit" onClick={deleteNode}><Trash2 size={14}/></button>
                </div>
            </div>

            <NodeViewContent className={`ghost-content content-div p-3 border-start border-end border-bottom rounded-bottom bg-white 
                ${node.attrs.linkedId ? 'border-primary-light' : 'border-purple-light'}
                ${node.attrs.isVisible ? 'border-solid' : 'border-dashed'}
            `} />

            <style>{`
                .border-purple { border-color: #8b5cf6 !important; }
                .bg-purple-subtle { background-color: #f5f3ff; color: #7c3aed; }
                .border-purple-light { border-color: rgba(139, 92, 246, 0.3) !important; }
                .border-primary-light { border-color: rgba(13, 110, 253, 0.3) !important; }
                .text-inherit { color: inherit; }
                .cursor-text { cursor: text; }
                .shadow-inset { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); }
                .w-40 { width: 40%; } .w-60 { width: 60%; }
                .ghost-content p { margin-bottom: 0.75rem; }
                .ghost-content p:last-child { margin-bottom: 0; }
                .border-solid { border-style: solid !important; }
                .border-dashed { border-style: dashed !important; }
                @media print { .ghost-section-wrapper { display: none; } }
            `}</style>
        </NodeViewWrapper>
    );
};

// --- MENTION COMPONENT ---
const MentionComponent = (props) => {
    const { node, updateAttributes, deleteNode } = props;
    const { label, context, isGhost, id } = node.attrs; 
    const { onPreview } = useContext(EditorActionContext);
    const [isOpen, setIsOpen] = useState(false);
    const triggerRef = useRef(null);
    const popupRef = useRef(null);

    const handlePreviewClick = () => {
        if (!onPreview || !id) return;
        onPreview(id, null);
    };

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (popupRef.current && !popupRef.current.contains(event.target) && !triggerRef.current.contains(event.target)) setIsOpen(false);
        };
        if (isOpen) document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [isOpen]);

    const getPopupStyle = () => {
        if (!triggerRef.current) return {};
        const rect = triggerRef.current.getBoundingClientRect();
        return { position: 'fixed', top: `${rect.bottom + 10}px`, left: `${rect.left}px`, zIndex: 10000, width: '280px' };
    };

    return (
        <NodeViewWrapper as="span" className={`mention-chip-wrapper ${isGhost ? 'ghost-mode' : ''}`}>
            <span ref={triggerRef} className={`mention-chip ${isGhost ? 'ghost' : ''} ${isOpen ? 'active' : ''}`} onClick={(e) => { e.stopPropagation(); setIsOpen(!isOpen); }}>
                {isGhost && <Ghost size={10} className="me-1 opacity-50" />}
                {label}
            </span>
            {isOpen && createPortal(
                <div ref={popupRef} className="bg-white border rounded-3 shadow-lg p-0 animate-fade-in text-start" style={getPopupStyle()}>
                    <div className="bg-light border-bottom px-3 py-2 d-flex justify-content-between align-items-center rounded-top-3">
                        <span className="tiny fw-bold text-uppercase text-muted tracking-wide">{context || "Reference"}</span>
                        <span className="badge bg-white border text-muted fw-normal" style={{fontSize: '0.6rem'}}>{isGhost ? 'Ghost' : 'Explicit'}</span>
                    </div>
                    <div className="p-3"><p className="small text-dark mb-0" style={{lineHeight: '1.4'}}>{label}</p></div>
                    <div className="px-3 py-2 border-top d-flex align-items-center justify-content-between bg-light bg-opacity-25 rounded-bottom-3">
                        <div className="d-flex gap-1">
                            <button className={`btn btn-sm d-flex align-items-center gap-2 px-2 rounded transition-all ${isGhost ? 'bg-dark text-white' : 'btn-white border text-muted'}`} onClick={() => updateAttributes({ isGhost: !isGhost })} title={isGhost ? "Make Explicit" : "Make Ghost Reference"}>
                                <Ghost size={14} />
                            </button>
                            <button className="btn btn-sm btn-white border text-primary hover-bg-primary-subtle px-2" onClick={handlePreviewClick} title="View Details">
                                <Eye size={14} />
                            </button>
                        </div>
                        <button className="btn btn-sm btn-white border text-danger hover-bg-danger-subtle px-2" onClick={deleteNode} title="Remove Reference"><Trash2 size={14} /></button>
                    </div>
                </div>,
                document.body
            )}
            <style>{`
                .mention-chip { background-color: #e7f1ff; color: #0d6efd; border-radius: 6px; padding: 2px 6px; margin: 0 2px; font-size: 0.9em; font-weight: 600; border: 1px solid rgba(13, 110, 253, 0.1); cursor: pointer; display: inline-flex; align-items: center; vertical-align: baseline; transition: all 0.2s ease; user-select: none; }
                .mention-chip:hover, .mention-chip.active { background-color: #d0e1fd; border-color: #86b7fe; }
                .mention-chip.ghost { background-color: transparent; color: #6c757d; border: 1px dashed #ced4da; }
                .mention-chip.ghost:hover, .mention-chip.ghost.active { border-color: #6c757d; background-color: #f8f9fa; }
            `}</style>
        </NodeViewWrapper>
    );
};

// --- MENTION LIST ---
const MentionList = React.forwardRef((props, ref) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selectItem = index => { const item = props.items[index]; if (item) props.command({ id: item.id, label: item.label, context: item.context }); };
  useEffect(() => setSelectedIndex(0), [props.items]);
  React.useImperativeHandle(ref, () => ({
    onKeyDown: ({ event }) => {
      if (event.key === 'ArrowUp') { setSelectedIndex((selectedIndex + props.items.length - 1) % props.items.length); return true; }
      if (event.key === 'ArrowDown') { setSelectedIndex((selectedIndex + 1) % props.items.length); return true; }
      if (event.key === 'Enter') { selectItem(selectedIndex); return true; }
      return false;
    },
  }));
  return (
    <div className="bg-white border rounded shadow-lg overflow-hidden" style={{minWidth: '200px', maxWidth: '300px'}}>
      {props.items.length ? props.items.map((item, index) => (
        <button key={index} className={`w-100 text-start btn btn-sm border-0 rounded-0 px-3 py-2 d-flex flex-column ${index === selectedIndex ? 'bg-primary text-white' : 'text-dark hover-bg-light'}`} onClick={() => selectItem(index)}>
           <span className="fw-bold small text-truncate w-100">{item.label}</span>
           {item.context && <span className={`small text-truncate w-100 ${index === selectedIndex ? 'text-white-50' : 'text-muted'}`} style={{fontSize: '0.7rem'}}>{item.context}</span>}
        </button>
      )) : <div className="p-2 text-muted small">No matches found</div>}
    </div>
  );
});

// --- TOOLBAR DROPDOWN ---
const ToolbarDropdown = ({ icon: Icon, label, tooltip, items, colorClass = "text-muted", onInsert }) => {
    const [isOpen, setIsOpen] = useState(false);
    const buttonRef = useRef(null);
    const [rect, setRect] = useState(null);
    const hasStrategyItems = items.some(i => i.isStrategy);

    return (
        <>
            <button 
                ref={buttonRef}
                onClick={() => { if (buttonRef.current) { setRect(buttonRef.current.getBoundingClientRect()); setIsOpen(!isOpen); } }}
                onMouseDown={(e) => e.preventDefault()} 
                className={`btn btn-sm d-flex align-items-center gap-2 px-2 py-1 transition-all position-relative ${isOpen ? 'bg-light-subtle shadow-inset' : 'hover-bg-white'}`}
                style={{border: '1px solid transparent', borderRadius: '6px', height: '32px', whiteSpace: 'nowrap'}}
            >
                <Icon size={14} className={colorClass} />
                <span className={`small fw-bold d-none d-xl-block ${colorClass}`}>{label}</span>
                {hasStrategyItems && <span className="position-absolute top-0 start-100 translate-middle p-1 bg-primary border border-light rounded-circle" style={{width: 8, height: 8}}></span>}
                <ChevronDown size={10} className="text-muted opacity-50"/>
            </button>
            <PortalDropdown isOpen={isOpen} rect={rect} onClose={() => setIsOpen(false)}>
                <div className="px-2 py-2 bg-light border-bottom sticky-top">
                    <span className="tiny fw-bold text-uppercase text-muted" style={{fontSize: '0.65rem', letterSpacing: '0.5px'}}>Insert {tooltip || label}</span>
                </div>
                {items.length === 0 ? <div className="p-3 text-center small text-muted fst-italic">No items found.</div> : items.map((item, idx) => (
                    <button key={idx} className="w-100 btn btn-sm text-start text-truncate hover-bg-light small d-flex align-items-center justify-content-between py-2 px-3 border-bottom border-light" onMouseDown={(e) => e.preventDefault()} onClick={() => { onInsert(item); setIsOpen(false); }}>
                        <div className="d-flex flex-column overflow-hidden me-2">
                            <span className="text-dark fw-medium text-truncate">{item.label}</span>
                            {item.detail && <span className="text-muted text-truncate" style={{fontSize: '0.7em'}}>{item.detail}</span>}
                        </div>
                        {item.isStrategy && <div className="bg-primary-subtle text-primary rounded-circle p-1"><div style={{width: 6, height: 6, borderRadius: '50%', backgroundColor: 'currentColor'}} /></div>}
                    </button>
                ))}
            </PortalDropdown>
        </>
    );
};


// ============================================================================
// 6. MAIN EDITOR
// ============================================================================
const RichTextEditor = ({ initialContent, onUpdate, placeholder, strategyArgs = [], cvCategories = {}, linkableItems = [], hints = [], onPreview }) => {
    const [evidenceStats, setEvidenceStats] = useState({ used: 0, total: 0 });
    const [previewContent, setPreviewContent] = useState(null);

    const calculateEvidenceStats = (doc, strategies) => {
        if (!doc || !strategies.length) return { used: 0, total: 0 };
        const strategyIds = new Set();
        strategies.forEach(arg => { arg.evidence.forEach(ev => strategyIds.add(ev.id)); });
        if (strategyIds.size === 0) return { used: 0, total: 0 };
        const usedIds = new Set();
        doc.descendants((node) => { if (node.type.name === 'mention') usedIds.add(node.attrs.id); });
        let count = 0;
        strategyIds.forEach(id => { if (usedIds.has(id)) count++; });
        return { used: count, total: strategyIds.size };
    };

    const allMentionItems = useMemo(() => {
        const combined = [];
        strategyArgs.forEach(arg => {
            combined.push({ id: `arg-${arg.id}`, label: arg.title, context: 'Strategy Argument' });
            arg.evidence.forEach(ev => combined.push({ id: ev.id, label: ev.label, context: `Evidence: ${arg.title}` }));
        });
        Object.entries(cvCategories).forEach(([cat, items]) => {
            items.forEach(i => combined.push({ id: i.id, label: i.label, context: cat })); 
        });
        return combined;
    }, [strategyArgs, cvCategories]);

    const editor = useEditor({
        extensions: [
            StarterKit.configure({ link: false }),
            Placeholder.configure({ placeholder: placeholder || 'Write using Codex...' }),
            Link.configure({ openOnClick: false }),
            GhostText, 
            GhostSection, 
            Mention.configure({
                HTMLAttributes: { class: 'mention-chip', 'data-label': 'Mention' },
                renderText({ options, node }) { return `${node.attrs.label ?? node.attrs.id}`; },
                renderHTML({ options, node }) { return ['span', options.HTMLAttributes, `${node.attrs.label ?? node.attrs.id}`] },
                suggestion: {
                    items: ({ query }) => allMentionItems.filter(i => i.label.toLowerCase().includes(query.toLowerCase())).slice(0, 10),
                    render: () => {
                        let component, popup;
                        return {
                            onStart: props => {
                                component = new ReactRenderer(MentionList, { props, editor: props.editor });
                                if (!props.clientRect) return;
                                popup = tippy('body', { getReferenceClientRect: props.clientRect, appendTo: () => document.body, content: component.element, showOnCreate: true, interactive: true, trigger: 'manual', placement: 'bottom-start' });
                            },
                            onUpdate: props => { component.updateProps(props); if (!props.clientRect) return; popup[0].setProps({ getReferenceClientRect: props.clientRect }); },
                            onKeyDown: props => { if (props.event.key === 'Escape') { popup[0].hide(); return true; } return component.ref?.onKeyDown(props); },
                            onExit: () => { popup[0].destroy(); component.destroy(); },
                        };
                    },
                },
            }).extend({
                addAttributes() {
                    return {
                        id: { default: null },
                        label: { default: null },
                        context: { default: null, parseHTML: el => el.getAttribute('data-context') },
                        isGhost: { default: false, parseHTML: element => element.getAttribute('data-ghost') === 'true', renderHTML: attributes => ({ 'data-ghost': attributes.isGhost }) },
                    };
                },
                addNodeView() { return ReactNodeViewRenderer(MentionComponent); },
            }),
        ],
        editorProps: { attributes: { class: 'ProseMirror', spellcheck: 'true' } },
        content: initialContent,
        onUpdate: ({ editor }) => { setEvidenceStats(calculateEvidenceStats(editor.state.doc, strategyArgs)); },
        onBlur: ({ editor }) => onUpdate(editor.getHTML()),
        onCreate: ({ editor }) => { setEvidenceStats(calculateEvidenceStats(editor.state.doc, strategyArgs)); },
    });

    const insertChip = (item) => editor?.chain().focus().insertContent({ type: 'mention', attrs: { id: item.id, label: item.label, context: item.context } }).insertContent(' ').run();
    
    const insertGhostSection = () => {
        editor?.chain().focus().insertContent({ 
            type: 'ghostSection', 
            attrs: { label: 'New Strategy Section' }, 
            content: [{ type: 'paragraph' }] 
        }).run();
    };

    const insertQuickLayout = () => {
        if (strategyArgs.length === 0) return;

        const instructions = {
            type: 'paragraph',
            content: [{ 
                type: 'text', 
                text: 'Quick Layout: Draft your response for each requirement below. Use the pre-filled evidence context as a guide.',
                marks: [{ type: 'italic' }, { type: 'ghostText' }]
            }]
        };

        const layoutNodes = strategyArgs.map(arg => {
            const sectionLabel = arg.requirementLabel || arg.title || 'New Section';
            const linkedId = arg.requirementId || null;
            const linkedType = arg.requirementId ? 'requirement' : null;

            const evidenceParagraphs = (arg.evidence && arg.evidence.length > 0) 
                ? arg.evidence.map(ev => ({
                    type: 'paragraph',
                    content: [
                        {
                            type: 'mention',
                            attrs: { id: ev.id, label: ev.label || 'Evidence', isGhost: true, context: 'Context' }
                        },
                        {
                            type: 'text',
                            text: ' ', 
                        },
                        {
                            type: 'text',
                            text: `(${ev.detail || ''})`, 
                            marks: [{ type: 'ghostText' }] 
                        },
                        {
                            type: 'text',
                            text: ': ',
                            marks: [{ type: 'ghostText' }] 
                        },
                        {
                            type: 'text',
                            text: ' ' 
                        }
                    ]
                }))
                : [{ type: 'paragraph' }]; 

            return {
                type: 'ghostSection',
                attrs: { 
                    label: sectionLabel, 
                    linkedId: linkedId, 
                    linkedType: linkedType,
                    isVisible: false 
                },
                content: evidenceParagraphs
            };
        });

        editor.chain().focus().insertContent([
            instructions, 
            ...layoutNodes,
            { type: 'paragraph' }
        ]).run();
    };

    // --- CLEAN HTML GENERATOR (Updated Hierarchy Logic) ---
    const generateCleanHTML = () => {
        if (!editor) return "";
        const json = editor.getJSON();

        // Map depth (from 1 to 4+) to the appropriate H tag and style
        const getHeadingData = (currentDepth) => {
            // H-Tag number is derived from the depth, capping at 4
            const tagNumber = Math.min(currentDepth, 4); 
            const tag = `h${tagNumber}`;
            
            // Define styles for a minimal, professional look (Times New Roman, 12pt base)
            let style = 'font-weight: bold; margin-bottom: 0.5em;';
            
            // Sizes get smaller as the hierarchy number increases (i.e., deeper nesting)
            switch (tag) {
                case 'h1': style += 'font-size: 14pt; margin-top: 1.6em;'; break; // Largest/Most Important
                case 'h2': style += 'font-size: 13pt; margin-top: 1.4em;'; break;
                case 'h3': style += 'font-size: 12pt; margin-top: 1.2em;'; break;
                case 'h4': 
                default:   style += 'font-size: 11pt; margin-top: 1em;'; break; // Smallest/Least Important
            }

            return { tag, style };
        };

        const processNode = (node, depth = 0) => {
            if (!node) return "";
            
            // 1. Handle Text
            if (node.type === "text") {
                if (node.marks && node.marks.some(m => m.type === "ghostText")) {
                    return ""; 
                }
                let text = node.text;
                if (node.marks) {
                    node.marks.forEach(m => {
                        if (m.type === 'bold') text = `<b>${text}</b>`;
                        if (m.type === 'italic') text = `<i>${text}</i>`;
                    });
                }
                return text;
            }

            // 2. Handle Ghost Sections 
            if (node.type === "ghostSection") {
                if (node.attrs && node.attrs.isVisible) {
                    const { tag, style } = getHeadingData(depth + 1); 
                    
                    // Recursively process children
                    const childrenHtml = node.content ? node.content.map(n => processNode(n, depth + 1)).join("") : "";
                    
                    return `<${tag} style="${style}">${node.attrs.label}</${tag}>${childrenHtml}`;
                }
                // If invisible, just process children's HTML (depth maintained)
                return node.content ? node.content.map(n => processNode(n, depth)).join("") : ""; 
            }

            // 3. Handle Mentions
            if (node.type === "mention") {
                if (node.attrs && node.attrs.isGhost) return ""; 
                return node.attrs.label || "";
            }

            // 4. Paragraphs
            if (node.type === "paragraph") {
                // Process children at the same block depth
                const childrenHtml = node.content ? node.content.map(n => processNode(n, depth)).join("") : "";
                if (!childrenHtml.trim()) return ""; 
                return `<p>${childrenHtml}</p>`;
            }
            
            // 5. Default traversal for lists, etc.
            if (node.content) {
                // Process children at the same block depth
                return node.content.map(n => processNode(n, depth)).join("");
            }
            return "";
        };

        // Process top-level content (depth 0)
        return (json.content || []).map(n => processNode(n, 0)).join("");
    };

    const handlePreview = () => {
        const cleanHtml = generateCleanHTML();
        setPreviewContent(cleanHtml);
    };

    const toggleGhostContext = () => {
        editor?.chain().focus().toggleMark('ghostText').run();
    };

    const ToolbarDivider = () => <div className="vr mx-2 opacity-25 h-50 my-auto flex-shrink-0"></div>;
    
    const ToolbarBtn = ({ action, icon: Icon, isActive, tooltip }) => {
        const btnRef = useRef(null);
        useEffect(() => { if(btnRef.current) tippy(btnRef.current, { content: tooltip, theme: 'light-border', animation: 'shift-away' }); }, [tooltip]);
        
        return (
            <button 
                ref={btnRef} 
                onClick={action} 
                onMouseDown={(e) => e.preventDefault()} 
                className={`btn btn-sm border-0 d-flex align-items-center justify-content-center flex-shrink-0 transition-all ${isActive ? 'text-purple bg-purple-subtle' : 'text-muted hover-text-dark hover-bg-light'}`} 
                style={{width: '32px', height: '32px', borderRadius: '6px'}} 
                type="button"
            >
                <Icon size={16} />
            </button>
        );
    };

    if (!editor) return null;

    const progress = evidenceStats.total > 0 ? Math.round((evidenceStats.used / evidenceStats.total) * 100) : 0;
    const progressColor = progress === 100 ? 'text-success' : progress > 50 ? 'text-primary' : 'text-muted';

    return (
        <EditorActionContext.Provider value={{ onPreview, cvCategories, linkableItems }}>
            <div className="d-flex flex-column h-100 border rounded-3 bg-white focus-within-shadow transition-all" style={{minHeight: '400px'}}>
                
                <PreviewSidePanel isOpen={previewContent !== null} onClose={() => setPreviewContent(null)} content={previewContent} />

                {hints.length > 0 && (
                    <div className="bg-blue-50 border-bottom px-3 py-2 d-flex align-items-start gap-2 animate-fade-in">
                        <Sparkles size={14} className="text-primary mt-1 flex-shrink-0" />
                        <div className="small text-dark flex-grow-1">
                            <span className="fw-bold text-primary me-1">AI Tip:</span> {hints[0]}
                        </div>
                        <button className="btn-close small" style={{fontSize: '0.5rem'}} onClick={() => {}}></button>
                    </div>
                )}

                {/* TOP TOOLBAR (ASSETS) */}
                <div className="d-flex align-items-center gap-2 p-2 border-bottom bg-light-subtle rounded-top-3 overflow-x-auto hide-scrollbar text-nowrap">
                    {strategyArgs.length > 0 && (
                        <div className="d-flex gap-2 align-items-center">
                            {strategyArgs.map(arg => (
                                <ToolbarDropdown key={arg.id} icon={Lightbulb} label={arg.title.length > 15 ? arg.title.substring(0, 15) + '...' : arg.title} tooltip={`Strategy: ${arg.title}`} items={arg.evidence} colorClass="text-primary" onInsert={insertChip} />
                            ))}
                            <ToolbarDivider />
                        </div>
                    )}
                    <ToolbarDropdown icon={Briefcase} label="Exp" tooltip="Experience" items={cvCategories.Experience || []} onInsert={insertChip} />
                    <ToolbarDropdown icon={GraduationCap} label="Edu" tooltip="Education" items={cvCategories.Education || []} onInsert={insertChip} />
                    <ToolbarDropdown icon={Cpu} label="Skills" tooltip="Skills" items={cvCategories.Skills || []} onInsert={insertChip} />
                    <ToolbarDropdown icon={Trophy} label="Achieve" tooltip="Achievements" items={cvCategories.Achievements || []} onInsert={insertChip} />
                    <ToolbarDropdown icon={Heart} label="Hobbies" tooltip="Hobbies" items={cvCategories.Hobbies || []} onInsert={insertChip} />
                </div>

                {/* BOTTOM TOOLBAR (CODEX + FORMATTING) */}
                <div className="d-flex align-items-center gap-1 px-2 py-1 border-bottom bg-white overflow-x-auto hide-scrollbar">
                    
                    <button 
                        className="btn btn-sm btn-white border shadow-sm text-primary d-flex align-items-center gap-2 px-3 me-2 hover-bg-primary-subtle" 
                        style={{color: '#0d6efd', borderColor: 'rgba(13, 110, 253, 0.3)'}}
                        onClick={handlePreview}
                        onMouseDown={(e) => e.preventDefault()} 
                        title="Preview Document (Clean)"
                    >
                        <Eye size={14} /> <span className="small fw-bold">Preview</span>
                    </button>
                    
                    <ToolbarDivider />

                    {/* Codex Actions */}
                    <button 
                        className="btn btn-sm btn-white border shadow-sm text-purple d-flex align-items-center gap-2 px-2 me-1 hover-bg-purple-subtle" 
                        onClick={insertGhostSection}
                        onMouseDown={(e) => e.preventDefault()} 
                        title="Insert Strategy Section (Type '>>> ')"
                        style={{color: '#7c3aed', borderColor: 'rgba(124, 58, 237, 0.3)'}}
                    >
                        <SquareDashedBottom size={14} /> <span className="small fw-bold d-none d-md-inline">Section</span>
                    </button>

                    <button 
                        className="btn btn-sm btn-white border shadow-sm text-purple d-flex align-items-center gap-2 px-2 me-2 hover-bg-purple-subtle" 
                        onClick={insertQuickLayout}
                        onMouseDown={(e) => e.preventDefault()} 
                        title="Quick Layout: Insert Requirement Sections"
                        style={{color: '#7c3aed', borderColor: 'rgba(124, 58, 237, 0.3)'}}
                    >
                        <LayoutTemplate size={14} /> <span className="small fw-bold d-none d-md-inline">Layout</span>
                    </button>

                    <ToolbarBtn 
                        icon={Ghost} 
                        action={toggleGhostContext} 
                        isActive={editor.isActive('ghostText')} 
                        tooltip="Ghost Context: Mark text as invisible context for AI" 
                    />
                    
                    <ToolbarDivider />

                    {/* Formatting */}
                    <ToolbarBtn icon={Bold} action={() => editor.chain().focus().toggleBold().run()} isActive={editor.isActive('bold')} tooltip="Bold" />
                    <ToolbarBtn icon={Italic} action={() => editor.chain().focus().toggleItalic().run()} isActive={editor.isActive('italic')} tooltip="Italic" />
                    <ToolbarBtn icon={List} action={() => editor.chain().focus().toggleBulletList().run()} isActive={editor.isActive('bulletList')} tooltip="Bullet List" />
                    <ToolbarBtn icon={ListOrdered} action={() => editor.chain().focus().toggleOrderedList().run()} isActive={editor.isActive('orderedList')} tooltip="Ordered List" />
                    <ToolbarBtn icon={Quote} action={() => editor.chain().focus().toggleBlockquote().run()} isActive={editor.isActive('blockquote')} tooltip="Blockquote" />
                    
                    <div className="ms-auto d-flex align-items-center gap-3 pe-2 flex-shrink-0">
                         {evidenceStats.total > 0 && (
                            <div className="d-flex align-items-center gap-2" title="Strategy Evidence Mentioned">
                                <div className="d-flex align-items-center position-relative" style={{width: 20, height: 20}}>
                                    <svg viewBox="0 0 36 36" className="circular-chart">
                                        <path className="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                        <path className={`circle ${progress === 100 ? 'stroke-success' : 'stroke-primary'}`} strokeDasharray={`${progress}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                    </svg>
                                </div>
                                <span className={`small fw-bold ${progressColor}`} style={{fontSize: '0.75rem'}}>{evidenceStats.used}/{evidenceStats.total}</span>
                            </div>
                        )}
                        <div className="d-flex align-items-center gap-1 text-muted small" style={{fontSize: '0.7rem'}}>
                            <Type size={12}/> {editor.storage.characterCount?.characters?.() || 0}
                        </div>
                    </div>
                </div>
                
                <EditorContent editor={editor} className="flex-grow-1 p-4" />
                
                <style>{`
                    .ProseMirror { outline: none; min-height: 100%; font-size: 1rem; line-height: 1.7; color: #333; }
                    .ProseMirror p.is-editor-empty:first-child::before { color: #adb5bd; content: attr(data-placeholder); float: left; height: 0; pointer-events: none; }
                    .focus-within-shadow:focus-within { box-shadow: 0 0 0 4px rgba(13, 110, 253, 0.1); border-color: #86b7fe !important; }
                    .hide-scrollbar::-webkit-scrollbar { display: none; } .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
                    .bg-blue-50 { background-color: #f0f7ff; }
                    
                    /* Ghost Mark Styling */
                    .ghost-text-mark { 
                        color: #6c757d; 
                        background-color: #f8f9fa; 
                        border-bottom: 2px dashed #ced4da;
                        opacity: 0.8;
                    }

                    .circular-chart { display: block; margin: 0 auto; max-width: 100%; max-height: 100%; }
                    .circle-bg { fill: none; stroke: #eee; stroke-width: 3.8; }
                    .circle { fill: none; stroke-width: 3.8; stroke-linecap: round; animation: progress 1s ease-out forwards; }
                    @keyframes progress { 0% { stroke-dasharray: 0 100; } }
                    .stroke-primary { stroke: var(--bs-primary); }
                    .stroke-success { stroke: var(--bs-success); }
                    
                    .hover-bg-purple-subtle:hover { background-color: #f5f3ff !important; }
                    .hover-bg-primary-subtle:hover { background-color: #cfe2ff !important; }
                    .text-purple { color: #7c3aed !important; }
                    .bg-purple-subtle { background-color: #f5f3ff !important; color: #7c3aed !important; }
                    .shadow-inset { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); }
                    .w-40 { width: 40%; } .w-60 { width: 60%; }
                    .ghost-content p { margin-bottom: 0.75rem; }
                    .ghost-content p:last-child { margin-bottom: 0; }
                    .border-solid { border-style: solid !important; }
                    .border-dashed { border-style: dashed !important; }
                    @media print { .ghost-section-wrapper { display: none; } }
                `}</style>
            </div>
        </EditorActionContext.Provider>
    );
};

export default RichTextEditor;