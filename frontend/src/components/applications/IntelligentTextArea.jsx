// frontend/src/components/applications/IntelligentTextArea.jsx
import React, { useState, useEffect, useRef, useMemo, forwardRef, useImperativeHandle } from 'react';

// --- CSS for the component ---
const STYLES = `
.intelligent-textarea-wrapper {
    position: relative;
    width: 100%;
}

/* --- Base button style --- */
.intelligent-textarea-btn {
    position: absolute;
    top: 0px;
    z-index: 5;
    border: none;
    background: var(--bs-light);
    color: var(--bs-primary);
    border-radius: 0 0.375rem 0 0.375rem;
    font-size: 1.1rem;
    width: 30px;
    height: 30px;
    line-height: 1;
    opacity: 0.6;
    transition: opacity 0.15s ease-in-out;
}
.intelligent-textarea-btn:hover {
    opacity: 1;
}

/* --- Maximize button is now the only one --- */
.intelligent-textarea-max-btn {
    right: 0px; /* Sits at the far right */
    font-size: 0.9rem; /* Smaller icon */
}
/* --- END MODIFIED --- */


.intelligent-editor {
    min-height: 80px;
    width: 100%;
    padding: 0.375rem 0.75rem;
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.5;
    color: var(--bs-body-color);
    background-color: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    overflow-y: auto;
    white-space: pre-wrap;
    /* --- MODIFIED: Padding for one button --- */
    padding-right: 35px;
}
.intelligent-editor:focus {
    color: var(--bs-body-color);
    background-color: var(--bs-body-bg);
    border-color: #86b7fe;
    outline: 0;
    box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
}
.intelligent-editor[data-placeholder="true"]:before {
    content: 'Click to add notes... Type @ to reference your CV...';
    color: var(--bs-gray-500);
    cursor: text;
    pointer-events: none;
}
.intelligent-editor a {
    color: var(--bs-primary);
    text-decoration: none;
    font-weight: 500;
    background-color: var(--bs-primary-bg-subtle);
    padding: 1px 4px;
    border-radius: 4px;
    cursor: pointer;
}
.suggestions-popup {
    position: absolute;
    z-index: 1050;
    background: var(--bs-white);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    box-shadow: var(--bs-box-shadow-lg);
    max-height: 250px;
    overflow-y: auto;
    width: 300px;
}
.suggestion-item {
    display: block;
    width: 100%;
    padding: 0.5rem 1rem;
    clear: both;
    font-weight: 400;
    color: var(--bs-body-color);
    text-align: inherit;
    text-decoration: none;
    white-space: nowrap;
    background-color: transparent;
    border: 0;
    cursor: pointer;
}
.suggestion-item:hover, .suggestion-item.active {
    background-color: var(--bs-primary-bg-subtle);
    color: var(--bs-primary-text-emphasis);
}
.suggestion-item small {
    display: block;
    color: var(--bs-text-muted);
}
`;

// Regex to find our custom references
const REFERENCE_REGEX = /\[(.*?)]<:(.*?)><(.*?)>/g;

const IntelligentTextArea = forwardRef(({ 
    initialValue, 
    onSave, 
    cv, 
    onShowPreview,
    onMaximize,
    manageSaveExternally = false,
    onLocalTextChange = () => {}
}, ref) => {
    const editorRef = useRef(null);
    const [rawText, setRawText] = useState(initialValue || '');
    const [isFocused, setIsFocused] = useState(false);
    
    const suggestionQueryRef = useRef(null);
    
    const [suggestions, setSuggestions] = useState({
        open: false,
        query: '',
        type: null,
        categoryType: null,
        items: [],
        position: { top: 0, left: 0 },
        activeIndex: 0,
    });

    const cvCategories = useMemo(() => {
        if (!cv) return [];
        return [
            { type: 'experiences', name: 'Experiences', items: cv.experiences || [] },
            { type: 'projects', name: 'Projects', items: cv.projects || [] },
            { type: 'skills', name: 'Skills', items: cv.skills || [] },
            { type: 'education', name: 'Education', items: cv.education || [] },
            { type: 'achievements', name: 'Achievements', items: cv.achievements || [] },
            { type: 'hobbies', name: 'Hobbies', items: cv.hobbies || [] },
        ];
    }, [cv]);

    // --- Helper Functions ---
    const getItemName = (item, type) => {
        switch(type) {
            case 'experiences': 
                return item.title;
            case 'projects': 
                return item.title || item.name;
            case 'skills': 
                return item.name;
            case 'education': 
                return item.degree;
            case 'achievements': 
                return item.text;
            case 'hobbies': 
                return item.name;
            default: 
                return 'Unknown';
        }
    }

    const getItemSubtitle = (item, type) => {
        switch(type) {
            case 'experiences': return item.company;
            case 'education': return item.institution;
            case 'projects': return null;
            case 'achievements': return item.context;
            default: return null;
        }
    }
    
    // --- Convert raw text to display HTML ---
    const parseRawToHtml = (text) => {
        if (!text) return '';
        return text.replace(REFERENCE_REGEX, (match, type, id, name) => {
            const el = document.createElement('a');
            el.href = "#";
            el.dataset.type = type;
            el.dataset.id = id;
            el.setAttribute("contenteditable", "false");
            el.textContent = name;
            return el.outerHTML;
        });
    };

    // --- Convert display HTML back to raw text ---
    const parseHtmlToRaw = (element) => {
        let raw = '';
        if (!element) return '';
        element.childNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE) {
                raw += node.textContent;
            } else if (node.nodeType === Node.ELEMENT_NODE && node.tagName === 'A') {
                const type = node.dataset.type;
                const id = node.dataset.id;
                const name = node.textContent;
                raw += `[${type}]<:${id}><${name}>`;
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                if (node.tagName === 'BR') {
                    raw += '\n';
                } else if (node.tagName === 'DIV') {
                    if (raw.length > 0 && !raw.endsWith('\n')) raw += '\n';
                    raw += parseHtmlToRaw(node);
                } else {
                    raw += parseHtmlToRaw(node);
                }
            }
        });
        return raw;
    };

    // --- Sync editor display when initialValue changes ---
    useEffect(() => {
        if (editorRef.current && !isFocused) {
            const newRawText = initialValue || '';
            setRawText(newRawText);
            editorRef.current.innerHTML = parseRawToHtml(newRawText);
            editorRef.current.dataset.placeholder = !newRawText;
        }
    }, [initialValue, isFocused]);

    // --- Handle editor blur (save) ---
    const handleBlur = () => {
        setIsFocused(false);
        setTimeout(() => {
             if (!isFocused) {
                setSuggestions(prev => ({ ...prev, open: false, type: null }));
             }
        }, 150);
        
        const currentRawText = parseHtmlToRaw(editorRef.current);
        setRawText(currentRawText);

        if (manageSaveExternally) {
            onLocalTextChange(currentRawText);
        } else if (currentRawText !== initialValue) {
            onSave(currentRawText);
        }
        
        if (editorRef.current) {
            editorRef.current.dataset.placeholder = !currentRawText.trim();
        }
    };

    // --- Handle editor focus ---
    const handleFocus = () => {
        setIsFocused(true);
        if (editorRef.current) {
            editorRef.current.dataset.placeholder = "false";
        }
    };

    // --- Get caret position (to place popup) ---
    const getCaretCoordinates = () => {
        const sel = window.getSelection();
        if (sel.rangeCount > 0) {
            const range = sel.getRangeAt(0).cloneRange();
            range.collapse(true);
            const span = document.createElement('span');
            range.insertNode(span);
            const rect = span.getBoundingClientRect();
            const { top, left } = rect;
            span.parentNode.removeChild(span);
            
            const editorRect = editorRef.current.getBoundingClientRect();
            
            if (top === 0 && left === 0) {
                return { top: editorRect.top + window.scrollY, left: editorRect.left + window.scrollX };
            }
            // --- FIX 2: This was `editorRect.top`, now it's `editorRect.left` ---
            return { top: rect.bottom - editorRect.top, left: rect.left - editorRect.left };
        }
        return { top: 0, left: 0 };
    };

    // --- Handle click on links ---
    const handleClick = (e) => {
        if (e.target.tagName === 'A' && e.target.dataset.type) {
            e.preventDefault();
            const { type, id } = e.target.dataset;
            const category = cvCategories.find(c => c.type === type);
            if (!category) return;
            
            const item = category.items.find(i => i.id === id);
            if (item) {
                onShowPreview(item, type);
            }
        }
    };

    // --- Main input handler ---
    const handleInput = () => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        
        const range = selection.getRangeAt(0);
        if (!range.collapsed) return;
        
        const node = range.startContainer;
        const offset = range.startOffset;

        if (editorRef.current) {
            const hasText = editorRef.current.textContent.trim().length > 0;
            editorRef.current.dataset.placeholder = !hasText;
        }
        
        if (!node || node.nodeType !== Node.TEXT_NODE) {
            setSuggestions(prev => ({ ...prev, open: false }));
            return;
        }

        const text = node.textContent.substring(0, offset);
        
        if (suggestions.type) {
            // --- We are *inside* a query (e.g., searching for items) ---
            const queryStartOffset = suggestionQueryRef.current?.offset || 0;

            // --- FIX 1: Add 1 to queryStartOffset to search *after* the @ ---
            const safeQueryStart = Math.min(queryStartOffset + 1, text.length);
            const queryText = text.substring(safeQueryStart).toLowerCase();
            
            const category = cvCategories.find(c => c.type === suggestions.categoryType);
            
            if (!category) {
                setSuggestions(prev => ({ ...prev, open: false, type: null }));
                return;
            }

            const items = category.items.filter(item => {
                const nameToSearch = getItemName(item, suggestions.categoryType);
                return nameToSearch.toLowerCase().includes(queryText);
            });
            
            setSuggestions(prev => ({
                ...prev,
                open: true,
                query: queryText,
                items: items,
                position: getCaretCoordinates(),
                activeIndex: 0,
            }));

        } else {
            // --- Check if we are *starting* a new query ---
            const atMatch = text.match(/@([\w\s]*)$/);
            if (atMatch) {
                const query = atMatch[1].toLowerCase();

                const items = cvCategories.filter(cat => 
                    cat.items.length > 0 && cat.name.toLowerCase().includes(query)
                );
                
                suggestionQueryRef.current = {
                    node: node,
                    offset: offset - query.length - 1 // This offset points AT the @
                };
                
                setSuggestions({
                    open: true,
                    query: query,
                    type: 'category',
                    categoryType: null,
                    items: items,
                    position: getCaretCoordinates(),
                    activeIndex: 0,
                });
            } else {
                setSuggestions(prev => ({ ...prev, open: false, type: null }));
            }
        }
    };

    // --- Insert reference ---
    const insertReference = (item, type) => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        
        if (!selection.focusNode) {
            console.error("Editor lost focus. Cannot insert reference.");
            return; 
        }

        const range = selection.getRangeAt(0);
        
        // --- FIX 1 (continued): This offset correctly points AT the @ ---
        const { node, offset } = suggestionQueryRef.current;
        
        // This will now delete the entire query, *including* the @
        range.setStart(node, offset); 
        range.setEnd(selection.focusNode, selection.focusOffset);
        range.deleteContents();

        const link = document.createElement('a');
        
        const itemName = getItemName(item, type);

        link.href = "#";
        link.dataset.type = type;
        link.dataset.id = item.id;
        link.setAttribute("contenteditable", "false");
        link.textContent = itemName;

        range.insertNode(link);
        range.setStartAfter(link);
        range.setEndAfter(link);
        selection.removeAllRanges();
        selection.addRange(range);

        const space = document.createTextNode(' ');
        range.insertNode(space);
        range.setStartAfter(space);
        range.setEndAfter(space);
        selection.removeAllRanges();
        selection.addRange(range);

        setSuggestions({ open: false, items: [], query: '', type: null, position: {}, activeIndex: 0 });
        editorRef.current.focus();
        handleInput();
    };

    // --- Handle suggestion selection ---
    const selectSuggestion = (item) => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;

        if (!selection.focusNode || !suggestionQueryRef.current) {
            return;
        }

        const range = selection.getRangeAt(0);

        if (suggestions.type === 'category') {
            // Stage 1: Category selected
            const category = cvCategories.find(c => c.type === item.type);
            const { node, offset } = suggestionQueryRef.current;

            // This deletes only the category query (e.g., "exp")
            const queryStartOffset = offset + 1;
            const queryEndOffset = queryStartOffset + suggestions.query.length;

            if (queryEndOffset <= node.textContent.length) {
                range.setStart(node, queryStartOffset);
                range.setEnd(node, queryEndOffset);
                range.deleteContents();
            } else {
                range.setStart(node, queryStartOffset);
                range.setEnd(selection.focusNode, selection.focusOffset);
                range.deleteContents();
            }
            
            // --- FIX 1 (continued): We NO LONGER update the offset ---
            // const newQueryRefOffset = offset + 1; // <-- REMOVED
            // suggestionQueryRef.current.offset = newQueryRefOffset; // <-- REMOVED
            // The offset will *stay* pointing at the @, which is what we want.

            setSuggestions(prev => ({
                ...prev,
                type: 'item',
                categoryType: item.type,
                query: '',
                items: category.items,
                activeIndex: 0,
                position: getCaretCoordinates(), // Get new position after deletion
            }));
            
        } else {
            // Stage 2: Item selected
            insertReference(item, suggestions.categoryType);
        }
    };

    // --- Keyboard navigation ---
    const handleKeyDown = (e) => {
        if (!suggestions.open) return;
        if (suggestions.items.length === 0) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSuggestions(prev => ({
                ...prev,
                activeIndex: (prev.activeIndex + 1) % prev.items.length,
            }));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSuggestions(prev => ({
                ...prev,
                activeIndex: (prev.activeIndex - 1 + prev.items.length) % prev.items.length,
            }));
        } else if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            selectSuggestion(suggestions.items[suggestions.activeIndex]);
        } else if (e.key === 'Backspace' && suggestions.query === '' && suggestions.type === 'item') {
             e.preventDefault();
             // --- FIX 1 (continued): No offset math needed here either ---
             setSuggestions(prev => ({
                ...prev,
                type: 'category',
                categoryType: null,
                query: '',
                items: cvCategories,
                activeIndex: 0,
             }));
             // suggestionQueryRef.current.offset = suggestionQueryRef.current.offset - 1; // <-- REMOVED
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setSuggestions(prev => ({ ...prev, open: false, type: null }));
        }
    };
    
    // --- REMOVED: handlePlusClick function ---

    // --- Expose functions to parent (for the modal) ---
    useImperativeHandle(ref, () => ({
        triggerCategorySearch: (categoryItem) => {
            editorRef.current.focus();
            
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(editorRef.current);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);

            const atNode = document.createTextNode('@');
            range.insertNode(atNode);
            
            // This offset points AT the @
            suggestionQueryRef.current = {
                node: atNode,
                offset: 0
            };
            
            const category = cvCategories.find(c => c.type === categoryItem.type);
            
            setSuggestions({
                open: true,
                type: 'item',
                categoryType: categoryItem.type,
                query: '',
                items: category?.items || [],
                position: getCaretCoordinates(),
                activeIndex: 0,
            });

            range.setStartAfter(atNode);
            range.setEndAfter(atNode);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }));

    // --- Render ---
    return (
        <div className="intelligent-textarea-wrapper">
            <style>{STYLES}</style>
            
            {!manageSaveExternally && (
                <>
                    {/* --- REMOVED: Plus Button --- */}
                    <button
                        type="button"
                        className="intelligent-textarea-btn intelligent-textarea-max-btn"
                        title="Maximize Editor"
                        onMouseDown={(e) => { e.preventDefault(); onMaximize(); }}
                    >
                        <i className="bi bi-arrows-fullscreen"></i>
                    </button>
                </>
            )}
            
            <div
                ref={editorRef}
                className="intelligent-editor"
                contentEditable="true"
                onFocus={handleFocus}
                onBlur={handleBlur}
                onInput={handleInput}
                onKeyDown={handleKeyDown}
                onClick={handleClick}
                data-placeholder={!rawText && !isFocused}
            />
            {suggestions.open && (
                <div
                    className="suggestions-popup"
                    style={{
                        top: suggestions.position.top + 5,
                        left: suggestions.position.left,
                        right: suggestions.position.right ? suggestions.position.right + 5 : 'auto'
                    }}
                >
                    {suggestions.items.length === 0 && (
                        <span className="suggestion-item text-muted">No results found</span>
                    )}
                    {suggestions.items.map((item, index) => {
                        const name = suggestions.type === 'item' ? getItemName(item, suggestions.categoryType) : item.name;
                        const subtitle = suggestions.type === 'item' ? getItemSubtitle(item, suggestions.categoryType) : null;
                        
                        return (
                            <button
                                key={item.id || item.type}
                                type="button"
                                className={`suggestion-item ${index === suggestions.activeIndex ? 'active' : ''}`}
                                onMouseDown={(e) => {
                                    e.preventDefault();
                                    selectSuggestion(item);
                                }}
                            >
                                {name}
                                {subtitle && <small>{subtitle}</small>}
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
});

export default IntelligentTextArea;