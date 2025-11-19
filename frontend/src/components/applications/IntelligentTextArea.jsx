// frontend/src/components/applications/IntelligentTextArea.jsx
import React, { useState, useEffect, useRef, useMemo, forwardRef, useImperativeHandle } from 'react';
import { createPortal } from 'react-dom';
import { 
    Briefcase, GraduationCap, Cpu, Heart, Trophy, FolderGit2, 
    Link as LinkIcon, ChevronRight, LayoutList, BrainCircuit, X, Globe 
} from 'lucide-react';

// --- ICONS MAPPING ---
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

// --- STYLES ---
const STYLES = `
.intelligent-editor {
    min-height: 80px; width: 100%; padding: 0.75rem;
    font-size: 0.9rem; line-height: 1.6;
    color: var(--bs-body-color); background-color: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color); border-radius: 0.5rem;
    overflow-y: auto; white-space: pre-wrap;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}
.intelligent-editor:focus {
    border-color: #86b7fe; outline: 0;
    box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
}
.intelligent-editor[data-placeholder="true"]:before {
    content: attr(data-placeholder-text);
    color: var(--bs-gray-400); cursor: text; pointer-events: none; font-style: normal; opacity: 0.8;
}
.intelligent-editor a {
    color: #0d6efd; text-decoration: none; font-weight: 600;
    background-color: rgba(13, 110, 253, 0.1); padding: 0 4px; border-radius: 4px;
    cursor: pointer; user-select: none; display: inline-block;
}
.intelligent-editor a.external-link {
    background-color: transparent; color: #0969da; text-decoration: underline; padding: 0; font-weight: 400;
}
.intelligent-editor a.external-link:hover { color: #0a58ca; background-color: rgba(13, 110, 253, 0.05); }

@keyframes slideUp { from { transform: translateY(100%); } to { transform: translateY(0); } }
.mobile-sheet { animation: slideUp 0.3s ease-out; border-radius: 1rem 1rem 0 0; border-bottom: none !important; }
.mobile-sheet .w-35 { width: 35%; } .mobile-sheet .w-65 { width: 65%; }
@media (min-width: 768px) { .menu-desktop { width: 500px; height: 320px; border-radius: 0.5rem; } .menu-desktop .w-35 { width: 35%; } .menu-desktop .w-65 { width: 65%; } }
`;

const REFERENCE_REGEX = /\[(.*?)]<:(.*?)><(.*?)>/g; 
const MARKDOWN_LINK_REGEX = /\[([^\]]+)\]\(([^)]+)\)/g; 

// --- RESPONSIVE MENU COMPONENT ---
const HierarchicalMenu = ({ categories, onSelect, position, onClose, searchQuery = "" }) => {
    // Filter logic: If searchQuery exists, filter items
    const filteredCategories = useMemo(() => {
        if (!searchQuery) return categories;
        const query = searchQuery.toLowerCase();
        
        return categories.map(cat => ({
            ...cat,
            items: cat.items.filter(item => {
                const label = item.label || item.name || item.title || "";
                const subtitle = item.subtitle || "";
                return label.toLowerCase().includes(query) || subtitle.toLowerCase().includes(query);
            })
        })).filter(c => c.items.length > 0);
    }, [categories, searchQuery]);

    const [activeCatId, setActiveCatId] = useState(null);
    
    // Auto-select first category if list changes
    useEffect(() => {
        if (filteredCategories.length > 0) {
            // If current active category is empty or gone, switch to first one
            if (!activeCatId || !filteredCategories.find(c => c.id === activeCatId)) {
                setActiveCatId(filteredCategories[0].id);
            }
        } else {
            setActiveCatId(null);
        }
    }, [filteredCategories]);

    const menuRef = useRef(null);
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) onClose();
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [onClose]);

    const style = useMemo(() => {
        if (isMobile) return { position: 'fixed', bottom: 0, left: 0, right: 0, height: '50vh', zIndex: 1060, display: 'flex' };
        const MENU_WIDTH = 500;
        const screenWidth = window.innerWidth;
        let left = position.left;
        if (left + MENU_WIDTH > screenWidth - 20) left = Math.max(10, (position.left + position.width) - MENU_WIDTH); 
        const top = Math.min(position.top + 8, window.innerHeight - 330);
        return { position: 'fixed', top, left, zIndex: 1060, display: 'flex' };
    }, [position, isMobile]);

    const activeItems = filteredCategories.find(c => c.id === activeCatId)?.items || [];

    return createPortal(
        <>
            {isMobile && <div className="modal-backdrop fade show" style={{zIndex: 1055}} onClick={onClose}></div>}
            <div ref={menuRef} className={`bg-white shadow-lg border overflow-hidden d-flex flex-column ${isMobile ? 'mobile-sheet' : 'menu-desktop'}`} style={style}>
                {isMobile && (
                    <div className="d-flex justify-content-between align-items-center px-3 py-2 border-bottom bg-light">
                        <span className="fw-bold small text-muted">Link Evidence</span>
                        <button className="btn btn-sm btn-icon" onClick={onClose}><X size={16}/></button>
                    </div>
                )}
                
                {/* Optional Search Header Display */}
                {searchQuery && (
                    <div className="bg-light border-bottom px-3 py-1 text-muted tiny fw-bold text-uppercase">
                        Filtering by: "{searchQuery}"
                    </div>
                )}

                <div className="d-flex flex-grow-1 overflow-hidden">
                    {/* LEFT PANE */}
                    <div className="w-35 border-end bg-light overflow-auto custom-scroll py-2 d-flex flex-column">
                        {filteredCategories.length === 0 ? (
                            <div className="p-3 text-center text-muted small fst-italic">No matches.</div>
                        ) : (
                            filteredCategories.map(cat => {
                                const Icon = ICONS[cat.type] || ICONS.default;
                                return (
                                    <button
                                        key={cat.id}
                                        className={`w-100 btn btn-sm text-start d-flex align-items-center justify-content-between px-3 py-2 border-0 rounded-0 transition-colors ${activeCatId === cat.id ? 'bg-white text-primary fw-bold shadow-inset' : 'text-muted hover-bg-white'}`}
                                        onMouseEnter={() => !isMobile && setActiveCatId(cat.id)}
                                        onClick={() => isMobile && setActiveCatId(cat.id)}
                                        onMouseDown={(e) => e.preventDefault()} 
                                    >
                                        <div className="d-flex align-items-center gap-2 text-truncate">
                                            <Icon size={14} className="flex-shrink-0" /> 
                                            <span className="text-truncate">{cat.name}</span>
                                        </div>
                                        {activeCatId === cat.id && !isMobile && <ChevronRight size={12} className="flex-shrink-0"/>}
                                    </button>
                                );
                            })
                        )}
                    </div>
                    {/* RIGHT PANE */}
                    <div className="w-65 overflow-auto custom-scroll bg-white">
                        <div className="sticky-top bg-white border-bottom px-3 py-2 text-muted tiny fw-bold text-uppercase tracking-wide">
                            {filteredCategories.find(c => c.id === activeCatId)?.name || "Items"}
                        </div>
                        {activeItems.length === 0 ? (
                            <div className="h-75 d-flex align-items-center justify-content-center text-muted small fst-italic">No items found.</div>
                        ) : (
                            activeItems.map(item => (
                                <button
                                    key={item.id}
                                    className="w-100 btn btn-sm text-start px-3 py-2 border-bottom border-light hover-bg-primary-subtle transition-colors"
                                    onClick={() => onSelect(item, filteredCategories.find(c => c.id === activeCatId).type)}
                                    onMouseDown={(e) => e.preventDefault()}
                                >
                                    <div className="d-flex flex-column">
                                        <span className="fw-bold text-dark small text-truncate">{item.label || item.name || item.title}</span>
                                        {item.subtitle && <span className="text-muted tiny text-truncate opacity-75" style={{fontSize:'0.75em'}}>{item.subtitle}</span>}
                                    </div>
                                </button>
                            ))
                        )}
                    </div>
                </div>
            </div>
            <style>{`.shadow-inset { box-shadow: inset 3px 0 0 var(--bs-primary); }`}</style>
        </>,
        document.body
    );
};

// --- MAIN EDITOR COMPONENT ---
const IntelligentTextArea = forwardRef(({ 
    initialValue, 
    onSave, 
    cv, 
    extraSuggestions = [], 
    onMention,
    placeholder = "Type @ to link evidence..."
}, ref) => {
    const editorRef = useRef(null);
    const [menuState, setMenuState] = useState({ open: false, position: { top: 0, left: 0, width: 0 }, mode: 'cursor', query: '' });
    const suggestionQueryRef = useRef(null);

    const categories = useMemo(() => {
        const cats = [];
        if (extraSuggestions.length > 0) cats.push({ id: 'mapped', type: 'evidence', name: 'Mapped Evidence', items: extraSuggestions });
        
        if (cv) {
            const toItems = (list, labelKey, subKey) => (list || []).map(i => ({ ...i, label: i[labelKey], subtitle: i[subKey] }));
            cats.push({ id: 'exp', type: 'experiences', name: 'Experience', items: toItems(cv.experiences, 'title', 'company') });
            cats.push({ id: 'proj', type: 'projects', name: 'Projects', items: toItems(cv.projects, 'title', 'description') });
            cats.push({ id: 'skill', type: 'skills', name: 'Skills', items: toItems(cv.skills, 'name', 'category') });
            cats.push({ id: 'ach', type: 'achievements', name: 'Achievements', items: toItems(cv.achievements, 'text', 'context') }); // Added Achievements
            cats.push({ id: 'edu', type: 'education', name: 'Education', items: toItems(cv.education, 'degree', 'institution') });
            cats.push({ id: 'hob', type: 'hobbies', name: 'Hobbies', items: toItems(cv.hobbies, 'name', '') });
        }
        return cats.filter(c => c.items.length > 0);
    }, [cv, extraSuggestions]);

    // ... (parseRawToHtml, parseHtmlToRaw, getCaretCoordinates, handleClick - KEEP SAME) ...
    const parseRawToHtml = (text) => text ? text.replace(REFERENCE_REGEX, (m, t, i, n) => `<a href="#" data-type="${t}" data-id="${i}" contenteditable="false">${n}</a>`).replace(MARKDOWN_LINK_REGEX, (m, l, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer" contenteditable="false" class="external-link">${l}</a>`) : '';
    const parseHtmlToRaw = (el) => { /* ... same logic ... */ 
        let raw = '';
        el.childNodes.forEach(node => {
            if (node.nodeType === 3) raw += node.textContent;
            else if (node.tagName === 'A') {
                if (node.dataset.type) raw += `[${node.dataset.type}]<:${node.dataset.id}><${node.textContent}>`;
                else raw += node.getAttribute('href') ? `[${node.textContent}](${node.getAttribute('href')})` : node.textContent;
            } else if (node.nodeType === 1) {
                if (node.tagName === 'BR') raw += '\n';
                else if (node.tagName === 'DIV') { if (raw.length>0 && !raw.endsWith('\n')) raw += '\n'; raw += parseHtmlToRaw(node); }
                else raw += parseHtmlToRaw(node);
            }
        });
        return raw;
    };
    const getCaretCoordinates = () => { /* ... same ... */ 
        const sel = window.getSelection();
        if (!sel.rangeCount) return { top: 0, left: 0, width: 0 };
        const range = sel.getRangeAt(0).cloneRange();
        range.collapse(true);
        const rect = range.getClientRects()[0];
        return rect ? { top: rect.bottom, left: rect.left, width: 0 } : { top: 0, left: 0, width: 0 };
    };
    const handleClick = (e) => {
        if (e.target.tagName === 'A' && e.target.classList.contains('external-link')) {
            const url = e.target.getAttribute('href');
            if (url) window.open(url, '_blank', 'noopener,noreferrer');
        }
    };

    // --- HANDLE INPUT (Updated for Query) ---
    const handleInput = () => {
        const sel = window.getSelection();
        if (!sel.rangeCount) return;
        const node = sel.anchorNode;
        if (node.nodeType !== 3) return;

        const text = node.textContent;
        const offset = sel.anchorOffset;
        const textBefore = text.slice(0, offset);
        
        const atIndex = textBefore.lastIndexOf('@');
        
        if (atIndex !== -1) {
            const query = textBefore.slice(atIndex + 1); // Get text after @
            suggestionQueryRef.current = { node, offset: atIndex }; // Store @ position
            setMenuState({ 
                open: true, 
                position: getCaretCoordinates(), 
                mode: 'cursor',
                query: query // Pass query to state
            });
        } else {
            setMenuState(prev => ({ ...prev, open: false, query: '' }));
        }
        
        if (editorRef.current) editorRef.current.dataset.placeholder = !editorRef.current.textContent.trim();
    };

    // --- INSERT ITEM (Updated) ---
    const insertItem = (item, type) => {
        if (menuState.mode === 'external') {
            setMenuState({ ...menuState, open: false, query: '' });
            if (onMention) onMention(item, type);
            return;
        }

        const sel = window.getSelection();
        const range = document.createRange();
        const { node, offset } = suggestionQueryRef.current; // Position of @

        // Delete everything from @ to current cursor (the query)
        // Note: This simplistic logic assumes the user hasn't moved the cursor elsewhere.
        // For robust implementation, we calculate the length of the query.
        const textContent = node.textContent;
        const currentCursor = sel.anchorOffset;
        
        range.setStart(node, offset);
        range.setEnd(node, currentCursor); // Delete up to cursor
        range.deleteContents();

        const a = document.createElement('a');
        a.href = "#";
        a.contentEditable = false;
        a.dataset.type = type;
        a.dataset.id = item.id;
        a.textContent = item.label || item.name;

        range.insertNode(a);
        range.setStartAfter(a);
        range.setEndAfter(a);
        
        const space = document.createTextNode('\u00A0');
        range.insertNode(space);
        range.setStartAfter(space);
        range.setEndAfter(space);
        
        sel.removeAllRanges();
        sel.addRange(range);

        setMenuState({ open: false, position: { top: 0, left: 0 }, mode: 'cursor', query: '' });
        editorRef.current.focus();
        
        if (onMention) onMention(item, type);
    };

    useEffect(() => {
        if (editorRef.current && (initialValue || editorRef.current.textContent.trim() === "")) {
             editorRef.current.innerHTML = parseRawToHtml(initialValue || "");
             editorRef.current.dataset.placeholder = !initialValue;
        }
    }, [initialValue]);

    const handleBlur = () => {
        if (editorRef.current) {
            const txt = parseHtmlToRaw(editorRef.current);
            if (txt !== initialValue) onSave(txt);
        }
    };

    useImperativeHandle(ref, () => ({
        openMenu: (rect) => {
            if (!rect) return;
            setMenuState({ 
                open: true, 
                position: { top: rect.bottom, left: rect.left, width: rect.width }, 
                mode: 'external',
                query: '' // Reset query for external open
            });
        },
        triggerSearch: () => {
            editorRef.current.focus();
        }
    }));

    return (
        <div className="intelligent-textarea-wrapper">
            <style>{STYLES}</style>
            <div 
                ref={editorRef}
                className="intelligent-editor custom-scroll"
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
                    onSelect={insertItem}
                    onClose={() => setMenuState({ ...menuState, open: false })}
                    searchQuery={menuState.query} // Pass query
                />
            )}
        </div>
    );
});

export default IntelligentTextArea;